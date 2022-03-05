#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:51:59 2020

@author: gpu3
"""

# -*- coding: utf-8 -*-

from __future__ import print_function

from os import path
from sys import modules
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pkg_resources import resource_filename
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from tensorflow.python.keras.preprocessing import image

from mnet_deep_cdr import Model_DiscSeg as DiscModel
from mnet_deep_cdr.mnet_utils import BW_img, disc_crop, mk_dir, files_with_ext

disc_list = [400, 500, 600, 700, 800]
DiscROI_size = 800
DiscSeg_size = 512
CDRSeg_size = 400

#data_type = '.jpg'
#parent_dir = os.path.dir('/home/gpu3/shubham/refuge/data/Tsegmentation/complete_refuge')
#data_img_path = path.abspath(path.join(parent_dir, 'data', 'REFUGE-Training400', 'Training400', 'Glaucoma'))
#label_img_path = path.abspath(path.join(parent_dir, 'data', 'Annotation-Training400',
#                                        'Annotation-Training400', 'Disc_Cup_Masks', 'Glaucoma'))


data_type = '.jpg'
parent_dir ='/home/gpu3/shubham/skin/train/'
data_img_path = path.abspath(path.join(parent_dir, 'image'))
label_img_path = path.abspath(path.join(parent_dir, 'mask'))



data_save_path = mk_dir(path.join(parent_dir, 'training_crop', 'data'))
label_save_path = mk_dir(path.join(parent_dir, 'training_crop', 'label'))

file_test_list = files_with_ext(data_img_path, data_type)

#DiscSeg_model = DiscModel.DeepModel(size_set=DiscSeg_size)
#DiscSeg_model.load_weights(path.join(parent_dir, 'deep_model', 'Model_DiscSeg_ORIGA.h5'))
#os.environ["SM_FRAMEWORK"] = "tf.keras"

#import segmentation_models as sm
#preprocess_input = sm.get_preprocessing('resnet50')
#DiscSeg_model = sm.Unet('efficientnetb4', input_shape=(512,512,3), classes=1, activation='sigmoid', encoder_weights=None)
#DiscSeg_model.load_weights('/home/gpu3/shubham/skin/pOSAL/segmentation/skin_efficientnetb4_segmentation.h5')
Disc_flat = None
temp_txt=file_test_list[0]
lineIdx=0
for lineIdx, temp_txt in enumerate(file_test_list):
    print('Processing Img {idx}: {temp_txt}'.format(idx=lineIdx + 1, temp_txt=temp_txt))
    # load image
    org_img = np.asarray(image.load_img(path.join(data_img_path, temp_txt)))
    # load label
    org_label = np.asarray(image.load_img(path.join(label_img_path, temp_txt[:-4] + '_segmentation.png')))[:, :, 0]
    new_label = np.zeros(np.shape(org_label) + (3,), dtype=np.uint8)
    new_label[org_label < 200, 0] = 255
    new_label[org_label < 100, 1] = 255

    # Disc region detection by U-Net
    temp_img = resize(org_img, (DiscSeg_size, DiscSeg_size, 3)) * 255
    temp_img = preprocess_input(temp_img)
    temp_img = temp_img/255
    temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
    disc_map = DiscSeg_model.predict([temp_img])
    a = disc_map
    disc_map = BW_img(np.reshape(disc_map, (DiscSeg_size, DiscSeg_size)), 0.5)

    regions = regionprops(label(disc_map))
    C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
    C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)
    disc_idx=1
    DiscROI_size=800
    for disc_idx, DiscROI_size in enumerate(disc_list):
        disc_region, err_coord, crop_coord = disc_crop(org_img, DiscROI_size, C_x, C_y)
        label_region, _, _ = disc_crop(new_label, DiscROI_size, C_x, C_y)
        Disc_flat = rotate(cv2.linearPolar(disc_region, (DiscROI_size / 2, DiscROI_size / 2), DiscROI_size / 2,
                                           cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)
        Label_flat = rotate(cv2.linearPolar(label_region, (DiscROI_size / 2, DiscROI_size / 2), DiscROI_size / 2,
                                            cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)

        disc_result = Image.fromarray((Disc_flat * 255).astype(np.uint8))
        filename = '{}_{}.png'.format(temp_txt[:-4], DiscROI_size)
        disc_result.save(path.join(data_save_path, filename))
        label_result = Image.fromarray((Label_flat * 255).astype(np.uint8))
        label_result.save(path.join(label_save_path, filename))

plt.imshow(Disc_flat)
plt.show()
