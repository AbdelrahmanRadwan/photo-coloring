import tensorflow as tf
import glob
from PIL import Image
import numpy
import numpy as np
import math
from skimage import io, color
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import misc
import cv2
import argparse

sess = tf.InteractiveSession() 


#region Intialize Data and Layers weights

num_of_video_frames = 0
AbColores_values = None
GreyImages_Batch = [] 
ColorImages_Batch = []
Batch_size = 1
CurrentBatch_indx = 1
EpochsNum = 200
ExamplesNum = 1    # Number of all Images in Db Dir
Imgsize = 224, 224 
GreyChannels = 1
ML_OUTPUT = None 
Fusion_output = None
 
Low_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,1,64], stddev=0.001),name="Low1"),
               'W_conv2':tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.001),name="Low2"),
               'W_conv3':tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.001),name="Low3"),
               'W_conv4':tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.001),name="Low4"), 
               'W_conv5':tf.Variable(tf.truncated_normal([3,3,256,256], stddev=0.001),name="Low5"),
               'W_conv6':tf.Variable(tf.truncated_normal([3,3,256,512], stddev=0.001),name="Low6")}
 
Low_biases = {'b_conv1':tf.Variable(tf.truncated_normal([64], stddev=0.001)),
              'b_conv2':tf.Variable(tf.truncated_normal([128], stddev=0.001)),
              'b_conv3':tf.Variable(tf.truncated_normal([128], stddev=0.001)),
              'b_conv4':tf.Variable(tf.truncated_normal([256], stddev=0.001)),
              'b_conv5':tf.Variable(tf.truncated_normal([256], stddev=0.001)),
              'b_conv6':tf.Variable(tf.truncated_normal([512], stddev=0.001))}
 
Mid_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
               'W_conv2':tf.Variable(tf.truncated_normal([3,3,512,256], stddev=0.001))}
 
Mid_biases = {'b_conv1':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
              'b_conv2':tf.Variable(tf.truncated_normal([256], stddev=0.001))}
 
Global_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
                  'W_conv2':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
                  'W_conv3':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001)),
                  'W_conv4':tf.Variable(tf.truncated_normal([3,3,512,512], stddev=0.001))}
 
Global_biases = {'b_conv1':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
                 'b_conv2':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
                 'b_conv3':tf.Variable(tf.truncated_normal([512], stddev=0.001)),
                 'b_conv4':tf.Variable(tf.truncated_normal([512], stddev=0.001))}
 
Color_weights = {'W_conv1':tf.Variable(tf.truncated_normal([3,3,512,256], stddev=0.001)),
                 'W_conv2':tf.Variable(tf.truncated_normal([3,3,256,128], stddev=0.001)),
                 'W_conv3':tf.Variable(tf.truncated_normal([3,3,128,64], stddev=0.001)),
                 'W_conv4':tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.001)),
                 'W_conv5':tf.Variable(tf.truncated_normal([3,3,64,32], stddev=0.001)),
                 'W_conv6':tf.Variable(tf.truncated_normal([3,3,32,2], stddev=0.001))}
 
Color_biases = {'b_conv1':tf.Variable(tf.truncated_normal([256], stddev=0.001)),
                    'b_conv2':tf.Variable(tf.truncated_normal([128], stddev=0.001)),
                    'b_conv3':tf.Variable(tf.truncated_normal([64], stddev=0.001)),
                    'b_conv4':tf.Variable(tf.truncated_normal([64], stddev=0.001)),
                    'b_conv5':tf.Variable(tf.truncated_normal([32], stddev=0.001)),
                    'b_conv6':tf.Variable(tf.truncated_normal([2], stddev=0.001))}    
 
#FuseModel = {"fusion_w": tf.Variable(tf.truncated_normal([512, 256], stddev=0.001)),"fusion_b": tf.Variable(tf.truncated_normal([256], stddev=0.001))}
 
MLnode_for_each_layer= [ 1024, 512, 256]
MLHidden_layer = []
MLHidden_layer.append({'weights': tf.Variable(tf.truncated_normal([7 * 7 * 512, MLnode_for_each_layer[0]], stddev=0.001)),
                     'biases': tf.Variable(tf.truncated_normal([MLnode_for_each_layer[0]],stddev=0.001))})
for i in range(1, 3):
    MLHidden_layer.append({'weights': tf.Variable(tf.truncated_normal([MLnode_for_each_layer[i - 1], MLnode_for_each_layer[i]], stddev=0.001)),
 
                         'biases': tf.Variable(tf.truncated_normal([MLnode_for_each_layer[i]], stddev=0.001))})
 
 #endregion
 
def Fusion_layer(MiddNetOutput, GlobalNetOutput,BatchSize):
    """A network that fuses the output of midd Net  and output of MLP(Global) Net
    together.
    Args:
    MiddNetOutput: Size of [?, H/8, W/8, 256].
    GlobalNetOutput: Size of [?,256]
    """
  
    GlobalNetOutput = tf.tile(GlobalNetOutput, [1, 28*28]) #ASK
    GlobalNetOutput = tf.reshape(GlobalNetOutput, [BatchSize, 28, 28, 256])
    Fusion_output = tf.concat([MiddNetOutput, GlobalNetOutput], 3)
    return Fusion_output

def ConstructML(input_tensor, layers_count, node_for_each_layer):
    """A fully connected Network MLP connected to Global Feature outputs
    Args:
    input_tensor : the output of global feature , shape = [ 1, 7, 7, 512]
    layer_count : number of layer in MLP actually = 3 or more
    node_for_each_layer : a list contains number of nodes in each layer [ 1024, 512, 256]
    """
    global   ML_OUTPUT 
 
    FeatureVector = tf.reshape(input_tensor,shape=[-1,7 * 7 * 512]) #ASK -1
    layers_output = tf.add(tf.matmul(FeatureVector, MLHidden_layer[0]['weights']), MLHidden_layer[0]['biases'])
    layers_output = tf.nn.relu(layers_output)
 
    for j in range(1, layers_count):
        layers_output = tf.add(tf.matmul(layers_output, MLHidden_layer[j]['weights']), MLHidden_layer[j]['biases'])
        layers_output = tf.nn.relu(layers_output)
 
    ML_OUTPUT = layers_output    
    return ML_OUTPUT

def Conv2d(inp, W ,Stride):
   """Computes a 2-D convolution over a 4-D input
    Args:
    inp: 4-D tensor
    W: Kernel
    Stride: The kenrel stride over the input
    """
   return tf.nn.conv2d(inp, W, strides=[1, Stride ,Stride, 1], padding='SAME')

def ReadNextBatch(): 
    '''
    Reads the Next (grey,Color) Batch and computes the Color_Images_batch Chrominance (AB colorspace values)
 
    Return:
     GreyImages_Batch: List with all Greyscale images [Batch size,224,224,1]
     ColorImages_Batch: List with all Colored images [Batch size,Colored images]
    '''
    global GreyImages_Batch
    global ColorImages_Batch
    global CurrentBatch_indx
    global Batch_size
    GreyImages_Batch = []
    ColorImages_Batch = []
    for ind in range(Batch_size):
        Colored_img = Image.open('test/' + str(CurrentBatch_indx) + '.jpg')
        ColorImages_Batch.append(Colored_img)
        Grey_img = Image.open('gray/' + str(CurrentBatch_indx) + '.jpg')
        Grey_img = np.asanyarray(Grey_img) 
        img_shape = Grey_img.shape
        img_reshaped = Grey_img.reshape(img_shape[0],img_shape[1], GreyChannels)#[224,224,1]
        GreyImages_Batch.append(img_reshaped)#[#imgs,224,224,1]
        CurrentBatch_indx = CurrentBatch_indx + 1
    Get_Batch_Chrominance() 
    return GreyImages_Batch

def Get_Batch_Chrominance():
    ''''
    Convert every image in the batch to LAB Colorspace and normalize each value of it between [0,1]
 
    Return:
     AbColores_values array [batch_size,224,224,2] 0 -> A value, 1 -> B value color
    '''
    global AbColores_values
    global ColorImages_Batch
    AbColores_values = np.empty((Batch_size,224,224,2),"float32")
    for indx in range(Batch_size):
        lab = color.rgb2lab(ColorImages_Batch[indx])
        Min_valueA = np.amin(lab[:,:,1])
        Max_valueA = np.amax(lab[:,:,1])
        Min_valueB = np.amin(lab[:,:,2])
        Max_valueB = np.amax(lab[:,:,2])
        AbColores_values[indx,:,:,0] = Normalize(lab[:,:,1],-128,128)
        AbColores_values[indx,:,:,1] = Normalize(lab[:,:,2],-128,128)

def Normalize(value,MinValue,MaxValue):
    '''Normalize the input value between specific range
 
    Args:
     value = pixel value
     MinValue = Old Min value
     MaxValue = Old Max value
 
   Return:
    Normalized Value
    '''
 
    MinNormalize_val = 0
    MaxNormalize_val = 1
    value = MinNormalize_val + (((MaxNormalize_val - MinNormalize_val) * (value - MinValue)) / (MaxValue - MinValue))
    return value

def DeNormalize(value,MinValue,MaxValue):
    '''DeNormalize the input value between specific range
 
    Args:
     value = pixel value
     MinValue = Old Min value
     MaxValue = Old Max value
 
   Return:
    Normalized Value
    ''' 
    MinNormalize_val = -128
    MaxNormalize_val = 128
    value = MinNormalize_val + (((MaxNormalize_val - MinNormalize_val) * (value - MinValue)) / (MaxValue - MinValue))
    return value

def Frobenius_Norm(M):
    '''Calculate Frobenius Normalization using formula Sqrt( sum(each (values^2) in input) )
 
    Args:
     M: Input Tensor     
    '''
    return tf.reduce_sum(M ** 2) ** 0.5

def TriainModel(Input):
    global Batch_size
 
    #region low level Net
    print("Intialize Low level Net")
 
 
    lowLev_layer1 = tf.nn.relu(Conv2d(Input, Low_weights['W_conv1'],2) + Low_biases['b_conv1']) 
    lowLev_layer2 = tf.nn.relu(Conv2d(lowLev_layer1, Low_weights['W_conv2'], 1) + Low_biases['b_conv2']) 
    lowLev_layer3 = tf.nn.relu(Conv2d(lowLev_layer2, Low_weights['W_conv3'], 2) + Low_biases['b_conv3']) 
    lowLev_layer4 = tf.nn.relu(Conv2d(lowLev_layer3, Low_weights['W_conv4'], 1) + Low_biases['b_conv4']) 
    lowLev_layer5 = tf.nn.relu(Conv2d(lowLev_layer4, Low_weights['W_conv5'], 2) + Low_biases['b_conv5']) 
    lowLev_layer6 = tf.nn.relu(Conv2d(lowLev_layer5, Low_weights['W_conv6'], 1) + Low_biases['b_conv6']) 
 
    #endregion
 
    #region Mid level Net
    print("--------------------------")
    print("Intialize Mid level Net")
 
 
    MidLev_layer1 = tf.nn.relu(Conv2d(lowLev_layer6, Mid_weights['W_conv1'], 1) + Mid_biases['b_conv1']) 
    MidLev_layer2 = tf.nn.relu(Conv2d(MidLev_layer1, Mid_weights['W_conv2'], 1) + Mid_biases['b_conv2']) 
 
    #endregion
 
    #region Global level Net
    print("--------------------------")
    print("Intialize Global Level Net")
 
    GlobalLev_layer1 = tf.nn.relu(Conv2d(lowLev_layer6, Global_weights['W_conv1'], 2) + Global_biases['b_conv1']) 
    GlobalLev_layer2 = tf.nn.relu(Conv2d(GlobalLev_layer1, Global_weights['W_conv2'], 1) + Global_biases['b_conv2']) 
    GlobalLev_layer3 = tf.nn.relu(Conv2d(GlobalLev_layer2, Global_weights['W_conv3'], 2) + Global_biases['b_conv3']) 
    GlobalLev_layer4 = tf.nn.relu(Conv2d(GlobalLev_layer3, Global_weights['W_conv4'], 1) + Global_biases['b_conv4']) 
 
    #endregion
 
    #region ML Net
    print("--------------------------")
    print("Intialize ML Net")
 
    ML_Net = ConstructML(GlobalLev_layer4,3,[1024,512,256])
    #endregion
 
    #region Fusion Layer
    print("--------------------------")
    print("initialize Fusion layer")   
    Fuse = Fusion_layer(MidLev_layer2, ML_OUTPUT,Batch_size)
    #endregion
 
    #region Colorization Net
    print("--------------------------")
    print("Intialize Colorization Net")
    print("--------------------------")
 
 
    Color_layer1 = tf.nn.relu(Conv2d(Fuse, Color_weights['W_conv1'], 1) + Color_biases['b_conv1'])     
 
    Color_layer2 = tf.nn.relu(Conv2d(Color_layer1, Color_weights['W_conv2'], 1) + Color_biases['b_conv2'])     
    Color_layer2_up = tf.image.resize_nearest_neighbor(Color_layer2,[56,56])
 
    Color_layer3 = tf.nn.relu(Conv2d(Color_layer2_up, Color_weights['W_conv3'], 1) + Color_biases['b_conv3']) 
    Color_layer4 = tf.nn.relu(Conv2d(Color_layer3, Color_weights['W_conv4'], 1) + Color_biases['b_conv4']) 
    Color_layer4_up =  tf.image.resize_nearest_neighbor(Color_layer4,[112,112])
 
    Color_layer5 = tf.nn.relu(Conv2d(Color_layer4_up, Color_weights['W_conv5'], 1) + Color_biases['b_conv5']) 
    Color_layer6 = tf.nn.sigmoid(Conv2d(Color_layer5, Color_weights['W_conv6'], 1) + Color_biases['b_conv6']) 
 
    Output =  tf.image.resize_nearest_neighbor(Color_layer6,[224,224])
 
    #endregion
 
    return Output

def TestModel(Input):
 
    #region low level Net
 
    lowLev_layer1 = tf.nn.relu(Conv2d(Input, Low_weights['W_conv1'],2) + Low_biases['b_conv1']) 
    lowLev_layer2 = tf.nn.relu(Conv2d(lowLev_layer1, Low_weights['W_conv2'], 1) + Low_biases['b_conv2']) 
    lowLev_layer3 = tf.nn.relu(Conv2d(lowLev_layer2, Low_weights['W_conv3'], 2) + Low_biases['b_conv3']) 
    lowLev_layer4 = tf.nn.relu(Conv2d(lowLev_layer3, Low_weights['W_conv4'], 1) + Low_biases['b_conv4']) 
    lowLev_layer5 = tf.nn.relu(Conv2d(lowLev_layer4, Low_weights['W_conv5'], 2) + Low_biases['b_conv5']) 
    lowLev_layer6 = tf.nn.relu(Conv2d(lowLev_layer5, Low_weights['W_conv6'], 1) + Low_biases['b_conv6']) 
 
    #endregion
 
    #region Mid level Net
 
    MidLev_layer1 = tf.nn.relu(Conv2d(lowLev_layer6, Mid_weights['W_conv1'], 1) + Mid_biases['b_conv1']) 
    MidLev_layer2 = tf.nn.relu(Conv2d(MidLev_layer1, Mid_weights['W_conv2'], 1) + Mid_biases['b_conv2']) 
 
    #endregion
 
    #region Global level Net
    
    GlobalLev_layer1 = tf.nn.relu(Conv2d(lowLev_layer6, Global_weights['W_conv1'], 2) + Global_biases['b_conv1']) 
    GlobalLev_layer2 = tf.nn.relu(Conv2d(GlobalLev_layer1, Global_weights['W_conv2'], 1) + Global_biases['b_conv2']) 
    GlobalLev_layer3 = tf.nn.relu(Conv2d(GlobalLev_layer2, Global_weights['W_conv3'], 2) + Global_biases['b_conv3']) 
    GlobalLev_layer4 = tf.nn.relu(Conv2d(GlobalLev_layer3, Global_weights['W_conv4'], 1) + Global_biases['b_conv4']) 
 
    #endregion
 
    #region ML Net

    ML_Net = ConstructML(GlobalLev_layer4,3,[1024,512,256])
    #endregion
 
    #region Fusion Layer
    Fuse = Fusion_layer(MidLev_layer2, ML_OUTPUT,1)
    #endregion
 
    #region Colorization Net
 
    Color_layer1 = tf.nn.relu(Conv2d(Fuse, Color_weights['W_conv1'], 1) + Color_biases['b_conv1'])     
 
    Color_layer2 = tf.nn.relu(Conv2d(Color_layer1, Color_weights['W_conv2'], 1) + Color_biases['b_conv2'])     
    Color_layer2_up = tf.image.resize_nearest_neighbor(Color_layer2,[56,56])
 
    Color_layer3 = tf.nn.relu(Conv2d(Color_layer2_up, Color_weights['W_conv3'], 1) + Color_biases['b_conv3']) 
    Color_layer4 = tf.nn.relu(Conv2d(Color_layer3, Color_weights['W_conv4'], 1) + Color_biases['b_conv4']) 
    Color_layer4_up =  tf.image.resize_nearest_neighbor(Color_layer4,[112,112])
 
    Color_layer5 = tf.nn.relu(Conv2d(Color_layer4_up, Color_weights['W_conv5'], 1) + Color_biases['b_conv5']) 
    Color_layer6 = tf.nn.sigmoid(Conv2d(Color_layer5, Color_weights['W_conv6'], 1) + Color_biases['b_conv6']) 
 
    Output =  tf.image.resize_nearest_neighbor(Color_layer6,[224,224])
 
 
    #endregion
 
    return Output

def Train():
    global AbColores_values
    global CurrentBatch_indx
    global GreyImages_Batch
    global EpochsNum
    global ExamplesNum
    global Batch_size
    Input_images = tf.placeholder(dtype=tf.float32,shape=[None,224,224,1],name="X_inputs")
    Ab_Labels_tensor = tf.placeholder(dtype=tf.float32,shape=[None,224,224,2],name="Labels_inputs")
    Prediction = TriainModel(Input_images) 
    Colorization_MSE = tf.reduce_mean((Frobenius_Norm(tf.subtract(Prediction,Ab_Labels_tensor)))) 
    Optmizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(Colorization_MSE)
    #sess.run(tf.global_variables_initializer()) 
    #saver = tf.train.Saver()
    #saver = tf.train.import_meta_graph('Model Directory/our_model.meta')
    #saver.restore(sess, 'Model Directory/our_model')

    for epoch in range(EpochsNum):
        epoch_loss = 0
        CurrentBatch_indx = 1
        for i in range(int(ExamplesNum / Batch_size)):#Over batches
           print("Batch Num ",i + 1)
           ReadNextBatch()
           a, c = sess.run([Optmizer,Colorization_MSE],feed_dict={Input_images:GreyImages_Batch,Ab_Labels_tensor:AbColores_values})
           epoch_loss += c
        print("epoch: ",epoch + 1, ",Loss: ",epoch_loss)

    #saver.save(sess, 'Model Directory/our_model',write_meta_graph=False)

def TestSingleImage():
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph('model/our_model.meta')
    saver.restore(sess, 'model/our_model')
    #region color
    GreyImages_Batch = []
    Grey_img = Image.open(r'1.jpg')
    Grey_img = np.asanyarray(Grey_img) 
    img_shape = Grey_img.shape
    img_reshaped = Grey_img.reshape(img_shape[0],img_shape[1], GreyChannels)#[224,224,1]
    GreyImages_Batch.append(img_reshaped)#[#imgs,224,224,1]
    TestImage = tf.placeholder(dtype=tf.float32,shape=[1,224,224,1])
    Prediction = TestModel(TestImage) 
    Chrominance = sess.run(Prediction,feed_dict={TestImage:GreyImages_Batch})
    NewImg=np.empty((224,224,3))
    for i in range(len(img_reshaped[:,1,0])):
        for j in range(len(img_reshaped[1,:,0])):
            NewImg[i,j,0]= 0 + ( (img_reshaped[i,j,0] - 0) * (100 - 0) / (255 - 0) )
    NewImg[:,:,1]=DeNormalize(Chrominance[0,:,:,0],0,1)
    NewImg[:,:,2]=DeNormalize(Chrominance[0,:,:,1],0,1)
    NewImg = color.lab2rgb(NewImg)
    #endregion
    #region plot
    plt.imshow(NewImg)
    plt.show()
    #endregion
    #region save
    plt.imsave(r'D:\FCIS\4th year materials\Graduation Project\Datasets\Places 365\Sub Output Images\300.jpg',NewImg)
    #endregion
    #region compare
    ground_truth=Image.open(r'D:\FCIS\4th year materials\Graduation Project\Datasets\Places 365\Sub Colored Images\300.jpg')
    ground_truth = ground_truth.convert('RGB')
    gray_ground_truth = ground_truth.convert('L')
    numpy_gray_ground_truth = np.asarray( gray_ground_truth, dtype="float" )
    numpy_gray_NewImg = color.rgb2gray(NewImg)
    Accuracy = compare_images(numpy_gray_NewImg, numpy_gray_ground_truth)
    #endregion
    print("Image Accuracy: ", Accuracy,"%\n")

def VideoToFrames():
    global num_of_video_frames

    vidcap = cv2.VideoCapture(r'D:\video.mp4')
    success,image = vidcap.read()
    success = True
    while success:
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        if success == False:
            break
        cv2.imwrite("D:\ColoredFrame%d.jpg" % num_of_video_frames, image)     # save colored frame 
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("D:\GrayFrame%d.jpg" % num_of_video_frames, image)        # save grayscale frame
        num_of_video_frames += 1
    print("Number of video frames : "+str(num_of_video_frames))

def Test():
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph('../model/our_model.meta')
    saver.restore(sess, '../model/our_model')

    for ind in range(1,2):
        GreyImages_Batch = []
        Grey_img = Image.open(r'../data/grey/1.jpg')

        Grey_img = Grey_img.convert("L")
        Grey_img = Grey_img.resize((224, 224), Image.NEAREST)  # use nearest neighbour
        print(Grey_img)

        Grey_img = np.asanyarray(Grey_img) 
        img_shape = Grey_img.shape
        img_reshaped = Grey_img.reshape(img_shape[0],img_shape[1], GreyChannels)#[224,224,1]
        GreyImages_Batch.append(img_reshaped)#[#imgs,224,224,1]
        TestImage = tf.placeholder(dtype=tf.float32,shape=[1,224,224,1])
        Prediction = TestModel(TestImage)
        print(Prediction)
        print("="*100)

        print(TestImage)
        print("="*100)
        print(GreyImages_Batch)
        Chrominance = sess.run(Prediction, feed_dict={TestImage:GreyImages_Batch})
        NewImg=np.empty((224,224,3))
        #NewImg[:,:,0]=img_reshaped[:,:,0]
        for i in range(len(img_reshaped[:,1,0])):
            for j in range(len(img_reshaped[1,:,0])):
                NewImg[i,j,0]= 0 + ( (img_reshaped[i,j,0] - 0) * (100 - 0) / (255 - 0) )
        #print(NewImg[:,:,0],"\n\n")
        NewImg[:,:,1]=DeNormalize(Chrominance[0,:,:,0],0,1)
        NewImg[:,:,2]=DeNormalize(Chrominance[0,:,:,1],0,1)
        #print(np.amin(NewImg),"\n\n")
        NewImg = color.lab2rgb(NewImg)
        #print("after RGB now: ",np.amin(NewImg),"\n\n")
        ground_truth_image =  Image.open(r'../data/colored/1.jpg')
        difference = 0.0
        ground_truth_image = ground_truth_image.convert('RGB')
        gray_ground_truth_image = ground_truth_image.convert('L')
        gray_NewImg = color.rgb2gray(NewImg)

        #for i in range(len(gray_NewImg[:,1])):
        #    for j in range(len(gray_NewImg[1,:])):
        #        print(gray_NewImg[i,j]," ",gray_ground_truth_image[i,j])

        #print(gray_ground_truth_image)
        gray_NewImg_as_image = Image.fromarray(gray_NewImg, 'L')
        #print(gray_NewImg_as_image)

        numpy_NewImg = np.asarray( gray_NewImg_as_image, dtype="float" )
        numpy_GroundTruth = np.asarray( gray_ground_truth_image, dtype="float" )

        Accuracy = compare_images(numpy_NewImg, numpy_GroundTruth)

        print("image "+ str(ind) +" Accuracy: ", Accuracy,"%")

        #print(numpy_GroundTruth)

        #print(gray_NewImg_as_image - gray_ground_truth_image)

        #print(np.sum(ground_truth_image)/(224*224))
        #print(np.average(ground_truth_image))
        plt.imsave(r'result.jpg',NewImg)
        print("image "+ str(ind) +" has been saved\n")

def TestVideo():
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph('Model Directory/our_model.meta')
    saver.restore(sess, 'Model Directory/our_model')

    for ind in range(1,101):  #(607)
        GreyImages_Batch = []                                                                    #gray frames\Frame' 
        Grey_img = Image.open(r'D:\FCIS\4th year materials\Graduation Project\Datasets\Places 365\Grey Images\\' + str(ind) + '.jpg')
        Grey_img = np.asanyarray(Grey_img) 
        img_shape = Grey_img.shape
        img_reshaped = Grey_img.reshape(img_shape[0],img_shape[1], GreyChannels)#[224,224,1]
        GreyImages_Batch.append(img_reshaped)#[#imgs,224,224,1]
        TestImage = tf.placeholder(dtype=tf.float32,shape=[1,224,224,1])
        Prediction = TestModel(TestImage) 
        Chrominance = sess.run(Prediction,feed_dict={TestImage:GreyImages_Batch})
        NewImg=np.empty((224,224,3))
        for i in range(len(img_reshaped[:,1,0])):
            for j in range(len(img_reshaped[1,:,0])):
                NewImg[i,j,0]= 0 + ( (img_reshaped[i,j,0] - 0) * (100 - 0) / (255 - 0) )
        NewImg[:,:,1]=DeNormalize(Chrominance[0,:,:,0],0,1)
        NewImg[:,:,2]=DeNormalize(Chrominance[0,:,:,1],0,1)
        NewImg = color.lab2rgb(NewImg)
        plt.imsave(r'D:\FCIS\4th year materials\Graduation Project\Datasets\Places 365\output frames/Frame' + str(ind) + '.jpg',NewImg)

def MakeVideo():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-ext", "--extension", required=False, default='jpg', help="extension name. default is 'jpg'.")
    ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
    args = vars(ap.parse_args())
    
    # Arguments
    dir_path = '.'
    ext = args['extension']
    output = args['output']
    
    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)
    
    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video',frame)
    height, width, channels = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))
    
    for image in images:
    
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)
    
        out.write(frame) # Write out frame to video
    
        cv2.imshow('video',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    
    print("The output video is {}".format(output))


def compare_images(img1, img2): 
    # returns the accuracy between two images

    difference = 0.0
    for i in range(len(img1[:,1])):
            for j in range(len(img1[1,:])):
                difference += abs(img1[i,j] - img2[i,j]) * 40
    accuracy = 100 - difference / ( 255 * 224 * 224 )

    return accuracy




#-------------------------------------------------------------------------------
#Train()
Test()
#TestVideo()
#MakeVideo()
#TestSingleImage()
#VideoToFrames()




