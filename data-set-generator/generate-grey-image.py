import glob
from PIL import Image
from os import rename , listdir
import tensorflow as tf
import numpy as np

def resize(folder_read, folder_save, size=(224, 224)):
    open_training = glob.glob('{}/*.jpg'.format(folder_read))
    save_training = {}
    for single_image in open_training:
        try:
            save_training[single_image] = single_image.split('/')[1]
        except:
            continue
    for single_image in save_training:
        img = Image.open (single_image)
        img = img.resize (size, Image.ANTIALIAS)
        img.save ('{}/{}'.format (folder_save, save_training[single_image]))


#resize('train/','test/')




def Gray(folder_read, folder_save):
    open_training = glob.glob('{}/*.jpg'.format(folder_read))
    save_training = {}
    for single_image in open_training:
        try:
            save_training[single_image] = single_image.split('/')[1]
        except:
            continue
    for single_image in save_training:
        img = Image.open (single_image)
        img = img.convert('L')

        img.save ('{}/{}'.format (folder_save, save_training[single_image]))

Gray('test/','gray/')


def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(224, 224, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict={X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype=np.float32)
    return X_flip




"""import os
os.getcwd()
collection = "test/"
for i, filename in enumerate(os.listdir(collection)):
    os.rename("test/" + filename, "test/" + str(i+1) + ".jpg")
"""