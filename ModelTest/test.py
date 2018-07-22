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



def Test(num_of_photos, names):
    saver = tf.train.Saver()
    saver = tf.train.import_meta_graph('../Models/Model.meta')
    saver.restore(sess, '../Models/Model')
    for i in range(0,num_of_photos):
        Feeder=[]
        Ground_Truth = Image.open(Test_Path+str(names[i])+'.jpg')
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
        Ground_Truth.show()
        plt.imshow(Colorized_Image)
        plt.show()
        plt.imsave('../data/Test-Data/predicted/'+'+.jpg',Colorized_Image)
        #Grey_Colorized_Image=color.rgb2gray(Colorized_Image)
        #Numpy_Grey_Colorized=np.asarray(Image.fromarray(Grey_Colorized_Image,'L'),dtype="float")
        #Numpy_Ground_Truth=np.asarray(Image.fromarray(Ground_Truth,'L'),dtype="float")
