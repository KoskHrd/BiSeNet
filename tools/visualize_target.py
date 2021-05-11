#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '.')
from configs import cfg_factory

from PIL import Image
import numpy as np
import  matplotlib.pyplot as plt
# import pandas as pd
import cv2

import os
from tqdm import tqdm

from lib import png


# # img_label = Image.open('C:/Users/55298/dataset/carla/dataA/dataA/CameraSeg/02_00_000.png')
# # img_label.show()
# label = np.asarray(Image.open('C:/Users/55298/dataset/carla/dataA/dataA/CameraSeg/02_00_000.png'))
# # print(label)
# # plt.imshow(label)
# # print(label.shape)

# label_gray = label[:,:,0]
# # print(label_gray)
# # print(label_gray.shape)
# plt.subplot(1,2,1)
# plt.imshow(label)
# plt.subplot(1,2,2)
# plt.imshow(label_gray)

# plt.show()

lookup_table = np.zeros((23,3), dtype='uint8')
### <https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera> ###
lookup_table = [
    [0,0,0],            # 0: unlabeled      分類なし
    [70,70,70],         # 1: building       建物
    [100,40,40],        # 2: fence          フェンス
    [55,90,80],         # 3: other          その他
    [220,20,60],        # 4: pedestrian     歩行者
    [153,153,153],      # 5: pole           ポール
    [157,234,50],       # 6: road line      道路線/白線/車線
    [128,64,128],       # 7: road           道路
    [244,35,232],       # 8: sidewalk       歩道
    [107,142,35],       # 9: vegetation     草木/植物/植生
    [0,0,142],          #10: vehicles       乗物
    [102,102,156],      #11: wall           壁
    [220,220,0],        #12: trafficsign    交通標識
    [70,130,180],       #13: sky            空
    [81,0,81],          #14: ground         地面
    [150,100,100],      #15: bridge         橋
    [230,150,140],      #16: rail track     線路
    [180,165,180],      #17: guard rail     ガードレール
    [250,170,30],       #18: traffic light  交通信号灯/信号機
    [110,190,160],      #19: static         静的物
    [170,120,50],       #20: dynamic        動的物
    [45,60,150],        #21: water          水
    [145,170,100]       #22: terrain        地形
    ]

### <https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge/discussion/101832> ###
# lookup_table = [
#     [0,0,0],            # 0: unlabeled      分類なし
#     [70,70,70],         # 1: building       建物
#     [190,153,153],      # 2: fence          フェンス
#     [250,170,160],      # 3: other          その他
#     [220,20,60],        # 4: pedestrian     歩行者
#     [153,153,153],      # 5: pole           ポール
#     [157,234,50],       # 6: road line      道路線/白線/車線
#     [128,64,128],       # 7: road           道路
#     [244,35,232],       # 8: sidewalk       歩道
#     [107,142,35],       # 9: vegetation     草木/植物/植生
#     [0,0,142],          #10: vehicles       乗物
#     [102,102,156],      #11: wall           壁
#     [220,220,0],        #12: trafficsign    交通標識
#     ]

# label_dst = label.copy()
# height,width,chn = label.shape
# for i in range(height):
#     for j in range(width):
#         label_dst[i,j,:] = lookup_table[label_gray[i,j]]
# # print(label_dst.shape)

# plt.subplot(1,2,1)
# plt.imshow(label)
# plt.subplot(1,2,2)
# plt.imshow(label_dst)

# plt.show()

cfg = cfg_factory['bisenetv2']

def img_concat_width(img1_name, img2_name):
    img1 = np.asarray(Image.open(img1_name))
    img2 = np.asarray(Image.open(img2_name))
    newImg = np.concatenate((img1, img2), axis=1)
    return Image.fromarray(newImg)

def cat():
    src1Path = './dataset/carla/dataABC/CameraRGB'
    src2Path = './res/vis/tar/dataABC'
    src1Name = os.path.join(src1Path, '02_00_000.png')
    src2Name = os.path.join(src2Path, '02_00_000.png')
    dstPath = './res/vis/tar'
    dstName = os.path.join(dstPath, 'dataA_001_edited.png')
    img_concat_width(src1Name, src2Name).save(dstName)


def main():
    src_path = cfg.val_img_anns
    dst_path = os.path.join(cfg.respth, 'vis/tar/dataD/')
    if not os.path.exists(dst_path): os.makedirs(dst_path)
    label_set = os.listdir(src_path)
    for k in tqdm(iterable=label_set, desc='To '+dst_path):
        # if k == "Thumbs.db":
        #     print(k)
        #     continue
        src_name = os.path.join(src_path, k)
        dst_name = os.path.join(dst_path, k)

        # label_binary = open(src_name, 'rb')
        # label_rawdata = label_binary.seek(number)
        # label_rawdata = label_binary.read()
        # label_binary.close()
        # label = Image.frombytes(mode='F', size=(600,800), data=label_rawdata,
        #                         decoder_name="raw", args='F;24B')
        # label_gray = np.asarray(label)

        label = np.asarray(Image.open(src_name))
        label_gray = label[:,:,0]

        # label = Image.open(src_name)
        # label = label.convert("RGB") if label.mode != "RGB" else label  # any format to RGB
        # label_gray = np.array(label)[:,:,0]

        # gamma22LUT = np.array([pow(x/255.0 , 2.2) for x in range(256)], dtype='float32')
        # label_bgr = cv2.imread(src_name)
        # label_bgrL = cv2.LUT(label_bgr, gamma22LUT)
        # label_grayL = label_bgrL[:,:,2]
        # label_gray = pow(label_grayL, 1.0/2.2) * 255

        # print("target.size(): {}".format(label_gray.shape))
        # print("target: {}".format(label_gray))
        # print("target.min(): {}".format(label_gray.min()))
        # print("target.max(): {}".format(label_gray.max()))
        # break

        label_dst = label.copy()
        height,width,_ = label.shape
        for i in range(height):
            for j in range(width):
                label_dst[i,j,:] = lookup_table[label_gray[i,j]]
        label_dst = Image.fromarray(label_dst)
        label_dst.save(dst_name)
        # plt.subplot(1,2,1)
        # plt.imshow(label)
        # plt.subplot(1,2,2)
        # plt.imshow(label_dst)
        # plt.show()

if __name__ == "__main__":
    # main()
    cat()
    pass