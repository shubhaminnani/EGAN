#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:21:49 2020

@author: gpu3
"""

'''
Created by SJWANG  07/27/2018
For refuge image segmentation
'''
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from multiprocessing import Pool, cpu_count
import timeit
from tqdm import tqdm
from skimage import transform
import cv2
import random
from Model.models import *
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
n_processes = cpu_count()
    # tspecify which GPU No. will you use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

t0 = timeit.default_timer()

''' parameter setting '''
DiscROI_size = 512
CDRSeg_size = 512
DiscSeg_size = 512
lr = 1e-4
dataset_t = "skin/"
dataset = "skin/"
models = []
phase = 'test'
data_type = '.jpg'
data_img_path = '/home/gpu3/shubham/skin/train/image_paper/'  # initial image path
data_save_path = '/home/gpu3/shubham/skin/train/predict_paper_e4_nogan/'  # save path

if not os.path.exists(data_save_path):
    print("Creating save path {}\n".format(data_save_path))
    os.makedirs(data_save_path)
#    file_test_list= file_test_list[50:]
file_test_list = [file for file in os.listdir(data_img_path) if file.lower().endswith(data_type)]
#random.shuffle(file_test_list)
print("==>[REFUGE challenge]\ttotal image number: {}\n".format(len(file_test_list)))
#file_test_list = file_test_list[534:]

weight_path = "./weights/generator_60.h5"
#    CDRSeg_weights_path.append("./weights/weights5.h5")
#CDRSegGAN_model = sm.Unet('efficientnetb7', input_shape=(512,512,3), classes=2, activation='sigmoid')
#CDRSegGAN_model.compile(optimizer=Adam(lr=lr), loss=Dice_Smooth_loss,metrics=[dice_coef_disc, dice_coef_cup, smooth_loss, dice_loss])

CDRSeg_model = sm.Unet('efficientnetb4', input_shape=(512,512,3), classes=1, activation='sigmoid')
CDRSeg_model.compile(optimizer=Adam(lr=lr), loss=Dice_Smooth_loss,metrics=[dice_coef_disc, dice_coef_cup, smooth_loss, dice_loss])
CDRSeg_model.load_weights('/home/gpu3/shubham/skin/segmentation/skin_efficientnetb4_segmentation.h5')
models.append(CDRSeg_model)

#
#    ''' define model '''
#    CDRSegGAN_model = Model_CupSeg(input_shape = (CDRSeg_size+32, CDRSeg_size+32, 3), classes=2,
#                                   backbone='mobilenetv2', lr=lr)
#
#    ''' whether to add initial segmentation model results'''
#    CDRSeg_model = Model_CupSeg(input_shape=(CDRSeg_size+32, CDRSeg_size+32, 3), classes=2, backbone='mobilenetv2')
#    for weight_path in CDRSeg_weights_path:
        CDRSeg_model = Model_CupSeg(input_shape=(CDRSeg_size + 32, CDRSeg_size + 32, 3), classes=2,
                                    backbone='mobilenetv2')
        CDRSeg_model.load_weights('./weights/mnv2.h5')
        models.append(CDRSeg_model)
#lineIdx = 0
def image_ds(k):
    ''' predict each image '''
#    for lineIdx in tqdm(ids):
#    
#    if k == n_processes - 1:
#        sub_file_test_list = file_test_list[k * int(len(file_test_list) / n_processes):]
#    else:
#        sub_file_test_list = file_test_list[k * int(len(file_test_list) / n_processes):
#                                            (k + 1) * int(len(file_test_list) / n_processes)]
#   li
    for lineIdx in tqdm(file_test_list):
        temp_txt = [elt.strip() for elt in lineIdx.split(',')]
        img_name = temp_txt[0][:-4] + '.png'
        org_img = np.asarray(image.load_img(data_img_path + temp_txt[0]))

        disc_region = cv2.resize(org_img,(512,512))
        final_mask = None
        scale=0
        for scale in range(1):
            img = disc_region       # [0-255]
            shape = img.shape
            if final_mask is None:
                final_mask = np.zeros((img.shape[0], img.shape[1], 2))
            if scale == 1:
                img = cv2.resize(img, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
            elif scale == 2:
                img = cv2.resize(img, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)

            x0 = 16
            y0 = 16
            x1 = 16
            y1 = 16
            if (img.shape[1] % 32) != 0:
                x0 = int((32 - img.shape[1] % 32) / 2)
                x1 = (32 - img.shape[1] % 32) - x0
                x0 += 16
                x1 += 16
            if (img.shape[0] % 32) != 0:
                y0 = int((32 - img.shape[0] % 32) / 2)
                y1 = (32 - img.shape[0] % 32) - y0
                y0 += 16
                y1 += 16
#            img0 = np.pad(img, ((y0, y1), (x0, x1), (0, 0)), 'symmetric')
            img0 = img
            inp0 = []
            inp1 = []
            for flip in range(2):
                for rot in range(4):
                    if flip > 0:
                        img = img0[::-1, ...]
                    else:
                        img = img0
                    if rot % 2 == 0:
                        inp0.append(np.rot90(img, k=rot))
                    else:
                        inp1.append(np.rot90(img, k=rot))

            inp0 = np.asarray(inp0)
            inp0 = imagenet_utils.preprocess_input(np.array(inp0, "float32"), mode='tf')
            inp1 = np.asarray(inp1)
            inp1 = imagenet_utils.preprocess_input(np.array(inp1, "float32"), mode='tf')

            mask = np.zeros((img0.shape[0], img0.shape[1], 2))
            for model in models:
                model=models[0]
                pred0 = model.predict(inp0, batch_size=1)
                pred1 = model.predict(inp1, batch_size=1)

                j = -1
                for flip in range(2):
                    for rot in range(4):
                        j += 1
                        if rot % 2 == 0:
                            pr = np.rot90(pred0[int(j / 2)], k=(4 - rot))
                        else:
                            pr = np.rot90(pred1[int(j / 2)], k=(4 - rot))
                        if flip > 0:
                            pr = pr[::-1, ...]
                        mask += pr  # [..., :2]

            mask /= (8 * len(models))
#            mask = mask[y0:mask.shape[0] - y1, x0:mask.shape[1] - x1, ...]
            if scale > 0:
                mask = cv2.resize(mask, (final_mask.shape[1], final_mask.shape[0]))
            final_mask += mask
        final_mask /= 1
        final_mask = final_mask*255
        dims = org_img.shape
        final_mask  = cv2.resize(final_mask,(dims[1],dims[0]))
        final_mask[final_mask>128]=255
        final_mask[final_mask<=128]=0
        final_mask = final_mask[:,:,0]
#        final_mask = 255 - final_mask
        cv2.imwrite(os.path.join(data_save_path,img_name),final_mask)
#        save_img(org_img, mask_path="NULL", data_save_path=data_save_path, img_name=img_name, prob_map=final_mask, err_coord=err_coord,
#                 crop_coord=crop_coord, DiscROI_size=DiscROI_size,
#                 org_img_size=org_img.shape, threshold=0.75, pt=False)

#    elapsed = timeit.default_timer() - t0
#    print('==>[REFUGE challenge]\tTime: {:.3f} min'.format(elapsed / 60))
if __name__ == "__main__":
    pool = Pool(processes=14)
    pool.map(image_ds, range(14))
    


