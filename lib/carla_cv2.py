#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
# import torch.distributed as dist
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

import sys
sys.path.insert(0, '.')
import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
from lib.base_dataset import BaseDataset, TransformationTrain, TransformationVal


# lookup_table = np.zeros((23,3), dtype='uint8')
# ### <https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera> ###
# lookup_table = [
#     [0,0,0],            # 0: unlabeled      分類なし
#     [70,70,70],         # 1: building       建物
#     [100,40,40],        # 2: fence          フェンス
#     [55,90,80],         # 3: other          その他
#     [220,20,60],        # 4: pedestrian     歩行者
#     [153,153,153],      # 5: pole           ポール
#     [157,234,50],       # 6: road line      道路線/白線/車線
#     [128,64,128],       # 7: road           道路
#     [244,35,232],       # 8: sidewalk       歩道
#     [107,142,35],       # 9: vegetation     草木/植物/植生
#     [0,0,142],          #10: vehicles       乗物
#     [102,102,156],      #11: wall           壁
#     [220,220,0],        #12: trafficsign    交通標識
#     [70,130,180],       #13: sky            空
#     [81,0,81],          #14: ground         地面
#     [150,100,100],      #15: bridge         橋
#     [230,150,140],      #16: rail track     線路
#     [180,165,180],      #17: guard rail     ガードレール
#     [250,170,30],       #18: traffic light  交通信号灯/信号機
#     [110,190,160],      #19: static         静的物
#     [170,120,50],       #20: dynamic        動的物
#     [45,60,150],        #21: water          水
#     [145,170,100]       #22: terrain        地形
#     ]

labels_info = [
    {"hasInstances": False,"driable": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 0},
    {"hasInstances": False,"driable": False, "category": "construction", "catid": 0, "name": "building", "ignoreInEval": True, "id": 1, "color": [70, 70, 70], "trainId": 1},
    {"hasInstances": False,"driable": False, "category": "construction", "catid": 0, "name": "fence", "ignoreInEval": True, "id": 2, "color": [100, 40, 40], "trainId": 2},
    {"hasInstances": False,"driable": False, "category": "object", "catid": 0, "name": "other", "ignoreInEval": True, "id": 3, "color": [55, 90, 80], "trainId": 3},
    {"hasInstances": False,"driable": False, "category": "human", "catid": 0, "name": "pedstrian", "ignoreInEval": True, "id": 4, "color": [220, 20, 60], "trainId": 4},
    {"hasInstances": False,"driable": False, "category": "object", "catid": 0, "name": "pole", "ignoreInEval": True, "id": 5, "color": [153, 153, 153], "trainId": 5},
    {"hasInstances": False,"driable": True, "category": "flat", "catid": 0, "name": "road line", "ignoreInEval": True, "id": 6, "color": [157, 234, 50], "trainId": 6},
    {"hasInstances": False,"driable": True, "category": "flat", "catid": 1, "name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 7},
    {"hasInstances": False,"driable": False, "category": "flat", "catid": 1, "name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244,35,232], "trainId": 8},
    {"hasInstances": False,"driable": False, "category": "nature", "catid": 1, "name": "vegetation", "ignoreInEval": True, "id": 9, "color": [107, 142, 35], "trainId": 9},
    {"hasInstances": False,"driable": False, "category": "vehicle", "catid": 1, "name": "vehicle", "ignoreInEval": True, "id": 10, "color": [0, 0, 142], "trainId": 10},
    {"hasInstances": False,"driable": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 11, "color": [102, 102, 156], "trainId": 11},
    {"hasInstances": False,"driable": False, "category": "object", "catid": 2, "name": "traffic sign", "ignoreInEval": False, "id": 12, "color": [220, 220, 0], "trainId": 12},
    {"hasInstances": False,"driable": False, "category": "sky", "catid": 2, "name": "sky", "ignoreInEval": False, "id": 13, "color": [70, 130, 180], "trainId": 255},
    {"hasInstances": False,"driable": False, "category": "flat", "catid": 2, "name": "ground", "ignoreInEval": True, "id": 14, "color": [81, 0, 81], "trainId": 255},
    {"hasInstances": False,"driable": False, "category": "construction", "catid": 2, "name": "bridge", "ignoreInEval": True, "id": 15, "color": [150, 100, 100], "trainId": 255},
    {"hasInstances": False,"driable": False, "category": "flat", "catid": 2, "name": "rail track", "ignoreInEval": True, "id": 16, "color": [230, 150, 140], "trainId": 255},
    {"hasInstances": False,"driable": False, "category": "object", "catid": 3, "name": "guard rail", "ignoreInEval": False, "id": 17, "color": [180, 165, 180], "trainId": 255},
    {"hasInstances": False,"driable": False, "category": "object", "catid": 3, "name": "traffic light", "ignoreInEval": True, "id": 18, "color": [250, 170, 30], "trainId": 255},
    {"hasInstances": False,"driable": False, "category": "void", "catid": 3, "name": "static", "ignoreInEval": False, "id": 19, "color": [110, 190, 160], "trainId": 255},
    {"hasInstances": False,"driable": False, "category": "void", "catid": 3, "name": "dynamic", "ignoreInEval": False, "id": 20, "color": [170, 120, 50], "trainId": 255},
    {"hasInstances": False,"driable": False, "category": "nature", "catid": 4, "name": "water", "ignoreInEval": False, "id": 21, "color": [45, 60, 150], "trainId": 255},
    {"hasInstances": False,"driable": False, "category": "nature", "catid": 4, "name": "terrain", "ignoreInEval": False, "id": 22, "color": [145, 170, 100], "trainId": 255},
]


class Carla(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, inputsize, tar_ignore=255, transforms=None, mode='train'):
        super(Carla, self).__init__(
                dataroot, annpath, transforms, mode)
        self.n_cats = 13  #23
        # self.tar_ignore = 255  #CHANGED: use "id" for training as it is.
        self.tar_ignore = tar_ignore  # not evaluation
        self.tar_map = np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.tar_map[el['id']] = el['trainId']
        self.to_tensor = T.ToTensor(
            # mean=(0.3257, 0.3690, 0.3223), # city, rgb
            # std=(0.2112, 0.2148, 0.2115),
            mean=(0.3524, 0.3581, 0.3521), # carla, rgb
            std=(0.2605, 0.2571, 0.2654),
            # #mean=[0.35245398366625774, 0.3581336873774504, 0.35212403845792456]
            # #std=[0.26050246625763057, 0.2570952503100638, 0.2654258674345489]
        )
        self.inputsize = inputsize


def get_data_loader(datapth, annpath, imgs_per_gpu, scales, cropsize, anns_ignore=255, max_iter=None, mode='train', distributed=False):
    if mode == 'train':
        transforms = TransformationTrain(scales, cropsize)
        batchsize = imgs_per_gpu
        shuffle = True
        drop_last = True
    elif mode == 'val':
        transforms = TransformationVal()
        batchsize = imgs_per_gpu
        shuffle = False
        drop_last = False

    ds = Carla(datapth, annpath, tar_ignore=anns_ignore, transforms=transforms, mode=mode)

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not max_iter is None
            n_train_imgs = imgs_per_gpu * dist.get_world_size() * max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
        )
    return dl

def prepare_data_loader(datapth, annpath, inputsize, imgs_per_gpu, gpu_count, scales, cropsize, anns_ignore=255, mode='train', distributed=False):
    if mode == 'train':
        transforms = TransformationTrain(scales, cropsize)
        batchsize = imgs_per_gpu * gpu_count
        shuffle = True
        drop_last = True
    elif mode == 'val':
        transforms = TransformationVal()
        batchsize = imgs_per_gpu * gpu_count
        shuffle = False
        drop_last = False
    elif mode == 'test':
        transforms = TransformationVal()
        batchsize = 1
        shuffle = False
        drop_last = False
    else: assert mode is None, "mode should be defined"

    ds = Carla(datapth, annpath, inputsize, tar_ignore=anns_ignore, transforms=transforms, mode=mode)

    dl = DataLoader(
        ds,
        batch_size=batchsize,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=4,
        pin_memory=True,
    )
    return dl



if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from configs import cfg_factory
    def parse_args():
        parse = argparse.ArgumentParser()
        parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
        # parse.add_argument('--port', dest='port', type=int, default=44554,)
        parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
        parse.add_argument('--finetune-from', type=str, default=None,)
        return parse.parse_args()
    args = parse_args()
    cfg = cfg_factory[args.model]
    ds = Carla(cfg.train_img_root,
            cfg.train_img_anns,
            cfg.input_size,
            mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
