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

num_of_video_frames = 0
AbColors_values = None
GreyImages_List = []
ColorImages_List = []
BatchSize = 1
BatchIdx = 1
Epochs = 100
ExamplesNum = 90000   # Number of all Images in Db Dir
Imgsize = 224, 224
GreyChannels = 1
ML_OUTPUT = None
Fusion_output = None
FC_Out=None
CImages_Path = '../../ttest-colored/'
GImages_Path = '../../ttest-grey/'
Test_Path = '../data/Test/test_grey/'
Ori_Path = '../data/Test/test_colored/'
idx = 1

sess=tf.InteractiveSession()

Low_Weight = {
    'wl1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.001)),
    'wl2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.001)),
    'wl3': tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.001)),
    'wl4': tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.001)),
    'wl5': tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.001)),
    'wl6': tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.001))
}

Low_Biases={
    'bl1':tf.Variable(tf.truncated_normal([64],stddev=0.001)),
    'bl2':tf.Variable(tf.truncated_normal([128],stddev=0.001)),
    'bl3':tf.Variable(tf.truncated_normal([128],stddev=0.001)),
    'bl4':tf.Variable(tf.truncated_normal([256],stddev=0.001)),
    'bl5':tf.Variable(tf.truncated_normal([256],stddev=0.001)),
    'bl6':tf.Variable(tf.truncated_normal([512],stddev=0.001))
}

Mid_Weight = {
    'wm1': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.001)),
    'wm2': tf.Variable(tf.truncated_normal([3, 3, 512, 256], stddev=0.001)),

}

Mid_Biases={
    'bm1':tf.Variable(tf.truncated_normal([512],stddev=0.001)),
    'bm2':tf.Variable(tf.truncated_normal([256],stddev=0.001)),

}


Global_Weight = {
    'wg1': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.001)),
    'wg2': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.001)),
    'wg3': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.001)),
    'wg4': tf.Variable(tf.truncated_normal([3, 3, 512, 512], stddev=0.001))
}


Global_Biases={
    'bg1':tf.Variable(tf.truncated_normal([512],stddev=0.001)),
    'bg2':tf.Variable(tf.truncated_normal([512],stddev=0.001)),
    'bg3':tf.Variable(tf.truncated_normal([512],stddev=0.001)),
    'bg4':tf.Variable(tf.truncated_normal([512],stddev=0.001)),

}

FC_Weight = {
    'wf1': tf.Variable(tf.truncated_normal([512*7*7,1024], stddev=0.001)),
    'wf2': tf.Variable(tf.truncated_normal([1024, 512], stddev=0.001)),
    'wf3': tf.Variable(tf.truncated_normal([512, 256], stddev=0.001)),

}


FC_Biases = {
    'bf1': tf.Variable(tf.truncated_normal([1024], stddev=0.001)),
    'bf2': tf.Variable(tf.truncated_normal([512], stddev=0.001)),
    'bf3': tf.Variable(tf.truncated_normal([256], stddev=0.001)),

}

ColorNet_Weight={

    'wc1': tf.Variable(tf.truncated_normal([3, 3, 512, 256], stddev=0.001)),
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 256, 128], stddev=0.001)),
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.001)),
    'wc4': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.001)),
    'wc5': tf.Variable(tf.truncated_normal([3, 3, 64, 32], stddev=0.001)),
    'wc6': tf.Variable(tf.truncated_normal([3, 3, 32, 2], stddev=0.001))

}

ColorNet_Biases={

    'bc1': tf.Variable(tf.truncated_normal([256], stddev=0.001)),
    'bc2': tf.Variable(tf.truncated_normal([128], stddev=0.001)),
    'bc3': tf.Variable(tf.truncated_normal([64], stddev=0.001)),
    'bc4': tf.Variable(tf.truncated_normal([64], stddev=0.001)),
    'bc5': tf.Variable(tf.truncated_normal([32], stddev=0.001)),
    'bc6': tf.Variable(tf.truncated_normal([2], stddev=0.001))

}



def Construct_FC(global_cnn_output):

    global FC_Out
    features = tf.reshape(global_cnn_output,shape=[-1,512*7*7])

    features = tf.add( tf.matmul(features,FC_Weight['wf1']),FC_Biases['bf1'])
    features = tf.nn.relu(features)

    features = tf.add(tf.matmul(features,FC_Weight['wf2']), FC_Biases['bf2'])
    features = tf.nn.relu(features)

    features = tf.add( tf.matmul(features,FC_Weight['wf3']) ,FC_Biases['bf3'])
    features = tf.nn.relu(features)

    FC_Out=features

    return features


def Construct_Fusion(mid_output,global_output):
    #print("constructing fusion started")
    global BatchSize
    global_output = tf.tile(global_output, [1, 28*28])
    global_output = tf.reshape(global_output, [BatchSize, 28, 28, 256])
    Fusion_output = tf.concat([mid_output, global_output], 3)
    #print("constructing fusion finished")
    return Fusion_output



def Get_Chrominance():
    #print("getting chromincance")
    global AbColors_values
    global ColorImages_List
    global BatchSize

    AbColors_values=np.empty((BatchSize, 224, 224, 2), "float32")

    for i in range (BatchSize):
        colored=color.rgb2lab(ColorImages_List[i])
        AbColors_values[i,:,:,0]=Normlization(colored[:,:,1],-128,128,0,1)
        AbColors_values[i,:,:,1]=Normlization(colored[:,:,2],-128,128,0,1)
    #print("Done with chormincace")


def Normlization(Value,MinVale,MaxValue,MinNormalizeValue,MaxNormalizeVale):
    '''
    normalize the Input
    :param value: pixl value
    :param MinVale:Old min Vale
    :param MaxValue: Old Max vale
    :return: Normailed Input between 0 1
    '''
    Value = MinNormalizeValue + (((MaxNormalizeVale-MinNormalizeValue)*(Value- MinVale))/(MaxValue-MinVale))
    return Value

def DeNormlization(Value,MinVale,MaxValue,MinNormalizeValue,MaxNormalizeVale ):
    '''

    :param Value:
    :param MinVale:
    :param MaxValue:
    :param MinNormalizeValue:
    :param MaxNormalizeVale:
    :return:
    '''
    Value = MinNormalizeValue + (((MaxNormalizeVale-MinNormalizeValue)*(Value- MinVale))/(MaxValue-MinVale))
    return Value

def F_Norm(Tens):
    return tf.reduce_sum(input_tensor=Tens**2)**0.5


def Construct_Graph(input):
    lowconv1 = tf.nn.relu(tf.nn.conv2d(input=input, filter=Low_Weight['wl1'], strides=[1, 2, 2, 1], padding='SAME') + Low_Biases['bl1'])
    lowconv2 = tf.nn.relu(tf.nn.conv2d(input=lowconv1, filter=Low_Weight['wl2'], strides=[1, 1, 1, 1], padding='SAME') + Low_Biases['bl2'])
    lowconv3 = tf.nn.relu(tf.nn.conv2d(input=lowconv2, filter=Low_Weight['wl3'], strides=[1, 2, 2, 1], padding='SAME') + Low_Biases['bl3'])
    lowconv4 = tf.nn.relu(tf.nn.conv2d(input=lowconv3, filter=Low_Weight['wl4'], strides=[1, 1, 1, 1], padding='SAME') + Low_Biases['bl4'])
    lowconv5 = tf.nn.relu(tf.nn.conv2d(input=lowconv4, filter=Low_Weight['wl5'], strides=[1, 2, 2, 1], padding='SAME') + Low_Biases['bl5'])
    lowconv6 = tf.nn.relu(tf.nn.conv2d(input=lowconv5, filter=Low_Weight['wl6'], strides=[1, 1, 1, 1], padding='SAME') + Low_Biases['bl6'])
    #Mid
    midconv1=tf.nn.relu(tf.nn.conv2d(input=lowconv6,filter=Mid_Weight['wm1'],strides=[1,1,1,1],padding='SAME')+Mid_Biases['bm1'])
    midconv2=tf.nn.relu(tf.nn.conv2d(input=midconv1,filter=Mid_Weight['wm2'],strides=[1,1,1,1],padding='SAME')+Mid_Biases['bm2'])


    #Global
    globalconv1=tf.nn.relu(tf.nn.conv2d(input=lowconv6, filter=Global_Weight['wg1'], strides=[1, 2, 2, 1], padding='SAME')+Global_Biases['bg1'])
    globalconv2=tf.nn.relu(tf.nn.conv2d(input=globalconv1, filter=Global_Weight['wg2'], strides=[1, 1, 1, 1], padding='SAME')+Global_Biases['bg2'])
    globalconv3=tf.nn.relu(tf.nn.conv2d(input=globalconv2, filter=Global_Weight['wg3'], strides=[1, 2, 2, 1], padding='SAME')+Global_Biases['bg3'])
    globalconv4=tf.nn.relu(tf.nn.conv2d(input=globalconv3, filter=Global_Weight['wg4'], strides=[1, 1, 1, 1], padding='SAME')+Global_Biases['bg4'])

    MM=Construct_FC(globalconv4)

    Fuse = Construct_Fusion(midconv2, FC_Out)

    colconv1 = tf.nn.relu(tf.nn.conv2d(input=Fuse,filter=ColorNet_Weight['wc1'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc1'])
    colconv2 = tf.nn.relu(tf.nn.conv2d(input=colconv1,filter=ColorNet_Weight['wc2'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc2'])
    colconv2_UpSample = tf.image.resize_nearest_neighbor(colconv2, [56, 56])
    colconv3 = tf.nn.relu(tf.nn.conv2d(input=colconv2_UpSample,filter=ColorNet_Weight['wc3'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc3'])
    colconv4 = tf.nn.relu(tf.nn.conv2d(input=colconv3,filter=ColorNet_Weight['wc4'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc4'])
    colconv4_UpSample = tf.image.resize_nearest_neighbor(colconv4,[112,112])
    colconv5 = tf.nn.relu(tf.nn.conv2d(input=colconv4_UpSample,filter=ColorNet_Weight['wc5'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc5'])
    colconv6 = tf.nn.relu(tf.nn.conv2d(input=colconv5,filter=ColorNet_Weight['wc6'],strides=[1,1,1,1],padding='SAME')+ColorNet_Biases['bc6'])

    output=tf.image.resize_nearest_neighbor(colconv6,[224,224])

    return  output

def Load_Batch():
    global GreyImages_List
    global ColorImages_List
    global idx
    global BatchSize
    GreyImages_List=[]
    ColorImages_List=[]
    diff=ExamplesNum-idx
    for i in range(min(BatchSize,diff+1)):
        Color = Image.open(CImages_Path+str(idx)+'.jpg')
        ColorImages_List.append(Color)
        Grey = Image.open(GImages_Path+str(idx)+'.jpg')
        Converted=np.asanyarray(Grey)
        Converted=Converted.reshape(np.shape(Converted)[0],np.shape(Converted)[1],1)
        GreyImages_List.append(Converted)
        idx=idx+1
    return GreyImages_List


def Training_Model():
    global AbColors_values
    global GreyImages_List
    global BatchSize
    global Epochs
    global ExamplesNum
    global idx

    Input = tf.placeholder(tf.float32, [None, 224, 224, 1])
    AB_Original = tf.placeholder(tf.float32, [None, 224, 224, 2])
    Prediction = Construct_Graph(Input)
    MSE = tf.reduce_mean(F_Norm(tf.subtract(Prediction, AB_Original)))
    optim = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(MSE)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver = tf.train.import_meta_graph('../Models/Model.meta')
    #saver.restore(sess, '../Models/Model')


    for i in range(Epochs):
        loss=0
        turns = int(ExamplesNum/BatchSize)
        idx=1
        for j in range(turns):
            print("Batch = ", j+1)
            Load_Batch()
            Get_Chrominance()
            a, c = sess.run([optim, MSE],feed_dict={Input:GreyImages_List, AB_Original:AbColors_values})
            loss+=c
        print("Epoch :", i+1,"Loss is: ",loss)
        saver.save(sess, "../Alex-Model/Model")

def Test(names):
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph('../Models/Model.meta')
    saver.restore(sess, '../Models/Model')
    for img in names:
        Feeder=[]
        Ground_Truth = Image.open(Test_Path+img+'.jpg')
        Test_Image = Ground_Truth.resize((224,224), Image.NEAREST)
        Test_Image = np.asanyarray(Test_Image)
        shape=Test_Image.shape
        Test_Image = Test_Image.reshape(shape[0], shape[1], 1)
        Feeder.append(Test_Image)
        Images_PlaceHoder=tf.placeholder(dtype=tf.float32,shape=[1,224,224,1])
        Output=Construct_Graph(Images_PlaceHoder)
        Colors = sess.run(Output, feed_dict={Images_PlaceHoder:Feeder})
        Colorized_Image=np.empty((224,224,3))
        for i in range(224):
            for j in range(224):
                Colorized_Image[i,j,0] = DeNormlization(Test_Image[i,j,0],0,255,0,100)
        Colorized_Image[:,:,1] = DeNormlization(Colors[0,:,:,0],0,1,-128,128)
        Colorized_Image[:,:,2]=DeNormlization(Colors[0,:,:,1],0,1,-128,128)
        Colorized_Image=color.lab2rgb(Colorized_Image)
        #Original = Image.open(Ori_Path+img+'.jpg')
        #Original.show()
        plt.imshow(Colorized_Image)
        plt.show()
        plt.imsave('../data/Test/predicted/'+str(img)+'.jpg',Colorized_Image)

#Training_Model()
Test(['ss'])

