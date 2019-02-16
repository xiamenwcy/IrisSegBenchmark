#coding:utf-8
#GAO Bin

from __future__ import print_function
import os,sys
import numpy as np
import cv2

images_path = '/data1/caiyong.wang/data/CASIA/train/SegmentationClass_1D/'
image_name = os.listdir(images_path)
#print(image_name)

n_background = 0
n_iris = 0


pixel_background = 0
pixel_iris = 0


def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:
        median = (data[size//2] + data[size//2-1])/2
        data[0] = median
    if size % 2 == 1:
        median = data[(size - 1) // 2]
        data[0] = median
    return data[0]


def count(img):
    global pixel_background,pixel_iris
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
                pixel_background += 1
           

            if img[i][j][0] == 1 and img[i][j][1] == 1 and img[i][j][2] == 1:
                pixel_iris += 1
            else:
                pass
    return pixel_background,pixel_iris

for i,im in enumerate(image_name):
    print(i)

    background_before = pixel_background
    iris_before = pixel_iris
   

    if im[-4:] == '.png':

        im = cv2.imread(images_path + im)
        print(im.shape)
        count(im)

    
        if background_before - pixel_background != 0:
            n_background += 1
        if iris_before - pixel_iris != 0:
            n_iris += 1
     
    else:
        pass

print('Summary')
print('n_background:',n_background) 
print('n_iris:',n_iris) 
print('before:',pixel_background)
print('before:',pixel_iris)
w, h, c = im.shape
print('w:',w)
print('h:',h)
#f(class) = frequency(class) / (image_count(class) * 480*360)

f_background = pixel_background*1.0/(n_background * w * h)
f_iris = pixel_iris*1.0/(n_iris * w * h)


median_f = [f_background,f_iris]

#weight(class) = median of f(class)) / f(class)
median = get_median(median_f)
weight_background = median/f_background
weight_iris = median/f_iris



print('weight_background:',weight_background)
print('weight_iris:',weight_iris)
