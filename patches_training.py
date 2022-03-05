from skimage import io
import os
from tqdm import tqdm
import numpy as np
import cv2
from glob import glob

img_files = sorted(glob('/home/gpu3/shubham/skin/Tsegmentation/complete/images/*'))
mask_files = sorted(glob('/home/gpu3/shubham/skin/Tsegmentation/complete/mask/*'))
i=0
for i in range(len(img_files)):
    assert img_files[i].split('/')[-1].split('.')[0]== mask_files[i].split('/')[-1].split('.')[0]

for i in tqdm(range(len(img_files))):
    areas=[]
    img = cv2.imread(img_files[i])
    img_name = img_files[i].split('/')[-1]
    dims = img.shape
    mask = cv2.imread(mask_files[i])
    mask[mask<200]=0
    mask[mask>200]=255
    mask = mask - 255
    mask = mask*255
    mask_name = mask_files[i].split('/')[-1]
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
#    img_area = dims[0]*dims[1]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    patch_img = img[y-10:y+h+10,x-10:x+w+10]
    patch_lbl = mask[y-10:y+h+10,x-10:x+w+10]    
    cv2.imwrite('/home/gpu3/shubham/skin/val_patch/image/'+img_name,patch_img)
    cv2.imwrite('/home/gpu3/shubham/skin/val_patch/mask/'+mask_name,patch_lbl)
