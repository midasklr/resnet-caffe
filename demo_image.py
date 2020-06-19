#coding=utf-8
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
from PIL import Image, ImageFont, ImageDraw # 导入模块
import cv2
import time
caffe.set_mode_gpu()
#指定网络结构 与 lenet_train_test.prototxt不同
MODEL_FILE = '/home/kong/caffe/ImageNet/resnet18/deploy.prototxt'
PRETRAINED = '/home/kong/caffe/ImageNet/resnet18/resnet18-solver_iter_1300000_0.691375.caffemodel'
#图片已经处理成 lenet.prototxt的输入要求（尺寸28x28）
IMAGE_FILE = '/home/kong/caffe/data/imagenet/val/ILSVRC2012_val_00000045.JPEG'
net = caffe.Classifier(MODEL_FILE, PRETRAINED)
mean_value = np.float32([104,117,123])
#mean_value = np.float32([108.639,107.734,113.387])
net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=mean_value)
input_h, input_w = (224, 224)

image = cv2.imread(IMAGE_FILE)
# mean_value = np.float32([126.19,133.485,141.46])
# 数据输入和分类
input_data = cv2.resize(image, (input_w, input_h))
input_data = input_data.transpose((2, 0, 1))
input_data = input_data - net.transformer.mean['data']
net.blobs['data'].data[0] = input_data
start = time.time()
net.forward()
outputs = net.blobs['prob'].data
print(outputs.argmax())
print( 'predicted class is: %d  conf: %f' %(outputs.argmax(),max(outputs[0])) )
#print( outputs )
# print('predicted class:%s  conf:%f' %(label_map[outputs.argmax()], outputs.max()))
print('inference cost: %f sec'%(time.time()-start))

