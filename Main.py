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


def Normlization(Value,MinVale,MaxValue):
    '''
    normalize the Input
    :param value: pixl value
    :param MinVale:Old min Vale
    :param MaxValue: Old Max vale
    :return: Normailed Input between 0 1
    '''
    MinNormalizeValue = 0
    MaxNormalizeVale = 1
    Value = MinNormalizeValue + (((MaxNormalizeVale-MinNormalizeValue)*(Value- MinVale))/(MaxValue-MinVale))
    return Value

def DeNormlization(Value,MinVale,MaxValue):
    '''
    normalize the Input
    :param value: pixl value
    :param MinVale:Old min Vale
    :param MaxValue: Old Max vale
    :return: Normailed Input between 128 -128
    '''
    MinNormalizeValue = -128
    MaxNormalizeVale = 128
    Value = MinNormalizeValue + (((MaxNormalizeVale-MinNormalizeValue)*(Value- MinVale))/(MaxValue-MinVale))
    return Value

def F_Norm(Tens):
    '''
    Caclutes the Frobenius Normalization using this Formula
    sqrt(sum(each values**2))
    #https://en.wikipedia.org/wiki/Frobenius_normal_form

    :param Tens: tf tensor
    :return: Frobenius Normalization of the tensor
    '''
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





def Construct_FC(global_cnn_output):

    features = tf.reshape(global_cnn_output,shape=[-1,512*7*7])
    # haneb2a negarrb 1

    features = tf.matmul(features,FC_Weight['wf1']) + FC_Biases['bf1']
    features=tf.nn.relu(features)

    features = tf.matmul(features,FC_Weight['wf2']) + FC_Biases['bf2']
    features=tf.nn.relu(features)

    features = tf.matmul(features,FC_Weight['wf3']) + FC_Biases['bf3']
    features=tf.nn.relu(features)

    return features


def Construct_Fusion(mid_output,global_output):
    global_output = tf.tile(global_output, [1, 28*28])
    global_output= tf.reshape(global_output, [BatchSize, 28, 28, 256])
    Fusion_output = tf.concat([mid_output, global_output], 3)
    return Fusion_output

