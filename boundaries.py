#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:29:04 2020

@author: gpu3
"""
import numpy as np
import os
from skimage import io
import cv2
from tqdm import tqdm
from skimage.segmentation import mark_boundaries
from glob import glob

img_path = '/home/gpu3/shubham/skin/train/paper/final/images'
mask_path = '/home/gpu3/shubham/skin/train/paper/final/mask_final'
predicted_path = '/home/gpu3/shubham/skin/train/paper/predict_paper_fpn/'
save_path = '/home/gpu3/shubham/skin/train/paper/final/pfpn_f'

imgs = sorted(glob(img_path+'/*'))
masks = sorted(glob(mask_path+'/*'))
preditions = sorted(glob(predicted_path+'/*'))
i=0
for i in tqdm(range(len(imgs))):
    fname = imgs[i].split('/')[-1].split('.')[0]+'.png'
    image = cv2.imread(imgs[i])/255
    image = cv2.resize(image, (256,256))
    label_img = cv2.imread(masks[i],0)/255
    label_img = cv2.resize(label_img,(256,256))
    predict_img = cv2.imread(predicted_path+fname,0)/255
    predict_img = cv2.resize(predict_img, (256,256))
    img_a = mark_boundaries(image, label_img, color=(0, 1, 0), mode={'thick', 'inner', 'outer'})
    img_b = mark_boundaries(img_a, predict_img, color=(0, 0, 1), mode={'thick', 'inner', 'outer'})
    img_b = img_b*255
    img_b = np.uint8(img_b)
    fname = preditions[i].split('/')[-1]
    cv2.imwrite(os.path.join(save_path,fname),img_b)

io.imshow(img_b)
save_path = '/home/gpu3/shubham/skin/train/paper/final'
s_path  = glob(save_path+'/*/*')

for i in tqdm(s_path):
    img = cv2.imread(i)
    img = cv2.resize(img,(512,512))
    folder_path =  i.split('ISIC')[0]
    fname = i.split('/')[-1]
    cv2.imwrite(folder_path+fname,img)
    