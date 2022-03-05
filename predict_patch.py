#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:45:01 2020

@author: gpu3
"""

'''
Created by SJWANG  07/27/2018
For refuge image segmentation
'''
from skimage import io
import timeit
from tqdm import tqdm
from skimage import transform
import cv2
import random
from Model.models import *
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

if __name__ == '__main__':
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
    data_img_path = '/home/gpu3/shubham/skin/val/image/'  # initial image path
    data_save_path = './results/segmentation/skin/test'  # save path
    data_mask_path = '/home/gpu3/shubham/skin/test_eff/'
    if not os.path.exists(data_save_path):
        print("Creating save path {}\n".format(data_save_path))
        os.makedirs(data_save_path)

    file_test_list = [file for file in os.listdir(data_img_path) if file.lower().endswith(data_type)]
    random.shuffle(file_test_list)
    print("==>[REFUGE challenge]\ttotal image number: {}\n".format(len(file_test_list)))


    ''' change to path '''
    DiscSeg_model_path = "./weights/Model_DiscSeg_pretrain.h5"
#    CDRSeg_weights_path = []
#    CDRSeg_weights_path.append("./weights/weights1.h5")
#    CDRSeg_weights_path.append("./weights/weights2.h5")
#    CDRSeg_weights_path.append("./weights/weights3.h5")
#    CDRSeg_weights_path.append("./weights/weights4.h5")
#    CDRSeg_weights_path.append("./weights/weights5.h5")
    CDRSeg_weights_path = "./weights/generator_10.h5"
    
    
    
    CDRSegGAN_model = sm.Unet('efficientnetb4', input_shape=(512,512,3), classes=2, activation='softmax')
    CDRSegGAN_model.compile(optimizer=Adam(lr=lr), loss=Dice_Smooth_loss,metrics=[dice_coef_disc, dice_coef_cup, smooth_loss, dice_loss])
    
    CDRSeg_model = sm.Unet('efficientnetb4', input_shape=(512,512,3), classes=2, activation='softmax')
    CDRSeg_model.compile(optimizer=Adam(lr=lr), loss=Dice_Smooth_loss,metrics=[dice_coef_disc, dice_coef_cup, smooth_loss, dice_loss])
    
    
    ''' create model and load weights'''
    DiscSeg_model = Model_DiscSeg(inputsize=DiscSeg_size)
    DiscSeg_model.load_weights(DiscSeg_model_path)

    ''' define model '''
    CDRSegGAN_model = Model_CupSeg(input_shape = (CDRSeg_size+32, CDRSeg_size+32, 3), classes=2,
                                   backbone='mobilenetv2', lr=lr)

    ''' whether to add initial segmentation model results'''
    CDRSeg_model = Model_CupSeg(input_shape=(CDRSeg_size+32, CDRSeg_size+32, 3), classes=2, backbone='mobilenetv2')
    for weight_path in CDRSeg_weights_path:
        CDRSeg_model = Model_CupSeg(input_shape=(CDRSeg_size + 32, CDRSeg_size + 32, 3), classes=2,
                                    backbone='mobilenetv2')
        CDRSeg_model.load_weights(CDRSeg_weights_path)
        models.append(CDRSeg_model)
lineIdx = 1

    ''' predict each image '''
    for lineIdx in tqdm(range(0, len(file_test_list))):
        temp_txt = [elt.strip() for elt in file_test_list[lineIdx].split(',')]
#        temp_txt = ['ISIC_0012650.jpg']
        img_name = temp_txt[0][:-4] + '.png'
#        if "refuge" in dataset:
#            img_name = temp_txt[0][:-4] + '.bmp'

        # if os.path.exists(os.path.join(data_save_path, img_name)):
        #     continue
        
        # load image
        org_img = np.asarray(image.load_img(data_img_path + temp_txt[0]))
        dims = org_img.shape
        mask = io.imread(data_mask_path+img_name)
#        mask_name = mask_files[i].split('/')[-1]
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        img_area = dims[0]*dims[1]
        max_index = np.argmax(areas)
        
        if areas!=[] and areas[max_index]<(0.6*img_area):
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            x,y,w,h = cv2.boundingRect(cnt)
            if y>20 and x>20:
                patch_img = org_img[y-10:y+h+10,x-10:x+w+10]
#                patch_lbl = mask[y-10:y+h+10,x-10:x+w+10]
            else:
                patch_img = org_img[y:y+h,x:x+w]
        else:
            patch_img = org_img
            x,y,w,h=0,0,patch_img.shape[0],patch_img.shape[1]
            
#                patch_lbl = mask[y:y+h,x:x+w]
            # Disc region detection by U-Net
#        temp_img = transform.resize(org_img, (DiscSeg_size, DiscSeg_size, 3)) * 255
#        temp_img = np.reshape(temp_img, (1,) + temp_img.shape)
#        [prob_10] = DiscSeg_model.predict([temp_img])
#
#        disc_map = BW_img(np.reshape(prob_10, (DiscSeg_size, DiscSeg_size)), 0.5)
#        regions = regionprops(label(disc_map))
#        C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size)
#        C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size)
#        
#        ''' get disc region'''
#        disc_region, err_coord, crop_coord = disc_crop(org_img, DiscROI_size, C_x, C_y)

        disc_region = cv2.resize(patch_img,(512,512))
        
        '''
        Test time augmentation
        '''
        final_mask = None
#        scale=0
#
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
            img0 = np.pad(img, ((y0, y1), (x0, x1), (0, 0)), 'symmetric')
#            img0 = img
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
            mask = mask[y0:mask.shape[0] - y1, x0:mask.shape[1] - x1, ...]
            if scale > 0:
                mask = cv2.resize(mask, (final_mask.shape[1], final_mask.shape[0]))
            final_mask += mask
            
        mask1 = cv2.resize(final_mask,(patch_img.shape[1],patch_img.shape[0]))
        mask_final = np.zeros([org_img.shape[0],org_img.shape[1]])
        if y>20 and x>20:
            mask_final[y-10:y+h+10,x-10:x+w+10] = mask1[:,:,0]
        elif x>0 and y>0 and x<21 and y<21:
            mask_final[y:y+h,x:x+w] = mask1[:,:,0]
        else:
            mask_final = mask1[:,:,0]
        
        mask_final = mask_final*255
        
#        mask_final  = cv2.resize(final_mask,(org_img.shape[1],org_img.shape[0]))
        mask_final[mask_final>128]=255
        mask_final[mask_final<=128]=0
#        mask_final = mask_final[:,:,0]
        mask_final = np.uint8(mask_final)
#        final_mask = 255 - final_mask
        cv2.imwrite(os.path.join(data_save_path,img_name),mask_final)
#        save_img(org_img, mask_path="NULL", data_save_path=data_save_path, img_name=img_name, prob_map=final_mask, err_coord=err_coord,
#                 crop_coord=crop_coord, DiscROI_size=DiscROI_size,
#                 org_img_size=org_img.shape, threshold=0.75, pt=False)

#    elapsed = timeit.default_timer() - t0
#    print('==>[REFUGE challenge]\tTime: {:.3f} min'.format(elapsed / 60))



