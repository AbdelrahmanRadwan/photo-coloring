from PIL import Image
import numpy as np
from skimage import color
from skimage import io
import cv2

a=Image.open("2.jpeg").convert('LA')
a=a.resize((244,244),Image.ANTIALIAS)
a=a.convert('L')
Grey_img = np.asanyarray(a)
print(np.shape(a))
img_shape = Grey_img.shape
img_reshaped = Grey_img.reshape(img_shape[0],img_shape[1], 1)#[224,224,1]
print(type(a),np.shape(a))
print(type(img_reshaped),np.shape(img_reshaped))
