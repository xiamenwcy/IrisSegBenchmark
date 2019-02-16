# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:46:35 2018

@author: wcy
"""
import numpy as np
import scipy.misc
import cv2
import scipy.io
import os, sys, argparse
import time
from os.path import join, splitext, split, isfile
parser = argparse.ArgumentParser(description='Forward all testing images.')
parser.add_argument('--model', type=str, default='../snapshot/nice/iris_fcn8s_iter_40000.caffemodel') 
parser.add_argument('--net', type=str, default='../model/nice/deploy.pt')
parser.add_argument('--gpu', type=int, default=5)
parser.add_argument('--ms', type=bool, default=True) # Using multiscale
args = parser.parse_args()
caffe_root = '/data1/caiyong.wang/program/deeplabv3/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
EPSILON = 1e-8

def forward(data):
  assert data.ndim == 3
  data -= np.array((104.00698793,116.66876762,122.67891434))
  data = data.transpose((2, 0, 1))
  net.blobs['data'].reshape(1, *data.shape)
  net.blobs['data'].data[...] = data
  return net.forward()

def create_labels_2(map): #2*h*w
    labels=np.argmax(map,axis=0).astype(np.float32)
    return labels

assert isfile(args.model) and isfile(args.net), 'file not exists'
USE_GPU = True
if USE_GPU:
   caffe.set_device(args.gpu)
   caffe.set_mode_gpu()
else:
   caffe.set_mode_cpu()

net = caffe.Net(args.net, args.model, caffe.TEST)
test_dir = '/data1/caiyong.wang/data/MASK/NICE/test/JPEGImages/' # test images directory
save_dir = join('/data1/caiyong.wang/program/IRIS_SUMMARY/cnn/FCN8s/test/nice/results_multi/', splitext(split(args.model)[1])[0]) # directory to save results
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
    
imgs = [i for i in os.listdir(test_dir) if '.JPEG' in i]
nimgs = len(imgs)
print "totally "+str(nimgs)+" images"
start = time.time()
for i in range(nimgs):
  img = imgs[i]
  img = cv2.imread(join(test_dir, img)).astype(np.float32)
  if img.ndim == 2:
    img = img[:, :, np.newaxis]
    img = np.repeat(img, 3, 2)
  
  h, w, _ = img.shape
  mask_bg = np.zeros((3,h, w), np.float32)
  mask_fg = np.zeros((3,h, w), np.float32)
  mask = np.zeros((2,h, w), np.float32)
  if args.ms:
    scales = [0.5, 1, 1.5]
  else:
    scales = [1]
  for j,s in enumerate(scales):
    h1, w1 = int(s * h), int(s * w)
    img1 = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    forward(img1)
    mask0 = np.squeeze( net.blobs['softmax_fuse_mask'].data[0][0,:,:])  #background
    mask1 = np.squeeze( net.blobs['softmax_fuse_mask'].data[0][1,:,:])   #foreground
    mask_bg[j,:,:]= cv2.resize(mask0, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mask_fg[j,:,:]= cv2.resize(mask1, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

  fn, ext = splitext(imgs[i])
 
  bg=np.max(mask_bg,axis=0).astype(np.float32)
  fg=np.max(mask_fg,axis=0).astype(np.float32)
  mask[0,:,:] = bg
  mask[1,:,:] = fg
  mask_out1 = create_labels_2(mask)

  
  fn, ext = splitext(imgs[i])
  scipy.misc.imsave(join(save_dir, fn+'.png'),mask_out1)
  print "Saving to '" + join(save_dir, imgs[i][0:-4]) + "', Processing %d of %d..."%(i + 1, nimgs)
end = time.time()
avg_time = (end-start)/nimgs
print("average time is %f seconds"%avg_time)