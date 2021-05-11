#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

# from PIL import Image

import sys
sys.path.insert(0, '.')
import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler


class BaseDataset(Dataset):
    def __init__(self, dataroot, annpath, transforms=None, mode='train'):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.transforms = transforms

        self.tar_map = None

        # with open(annpath, 'r') as fr:
        #     pairs = fr.read().splitlines()
        # self.img_paths, self.tar_paths = [], []
        # for pair in pairs:
        #     imgpth, tarpth = pair.split(',')
        #     self.img_paths.append(os.path.join(dataroot, imgpth))
        #     self.tar_paths.append(os.path.join(dataroot, tarpth))
        name_list = os.listdir(annpath)
        self.img_paths, self.tar_paths = [], []
        for name in name_list:
            self.img_paths.append(os.path.join(dataroot, name))
            self.tar_paths.append(os.path.join(annpath, name))

        assert len(self.img_paths) == len(self.tar_paths)
        self.len = len(self.img_paths)

    def __getitem__(self, idx):
        impth, tarpth = self.img_paths[idx], self.tar_paths[idx]
        # image, target = cv2.imread(impth), cv2.imread(tarpth, 0)
        # image, target = cv2.imread(impth), cv2.cvtColor(cv2.imread(tarpth), cv2.COLOR_BGR2GRAY)
        image, target = cv2.imread(impth), cv2.imread(tarpth)[:,:,2]
        # image, target = np.asarray(Image.open(impth)), np.asarray(Image.open(tarpth))[:,:,0]
        if not self.tar_map is None:
            target = self.tar_map[target]
        if not self.inputsize is None:
            image = cv2.resize(image, (self.inputsize[1], self.inputsize[0]))
            target = cv2.resize(target, (self.inputsize[1], self.inputsize[0]), interpolation=cv2.INTER_NEAREST)
        # #NoCHANGED: target.shape (H,W)>>(n_classes,H,W) by one-hot encoding.
        # ## https://campus.datacamp.com/courses/image-processing-with-keras-in-python/image-processing-with-neural-networks?ex=5
        # # The number of image categories
        # n_categories = 23
        # # The unique values of categories in the data
        # categories = np.array(np.arange(n_categories))
        # # Initialize ohe_labels as all zeros
        # ohe_target = np.zeros((target.shape[0], target.shape[1], n_categories))
        # # Loop over the labels
        # for ii in categories:
        #     # Find the location of this label in the categories variable
        #     jj = np.where(target==ii)
        #     # Set the corresponding zero to one
        #     ohe_target[jj[0],jj[1],ii] = 1
        img_tar = dict(img=image, tar=target)
        # img_tar = dict(img=image, tar=ohe_target)
        if not self.transforms is None:
            img_tar = self.transforms(img_tar)
        img_tar = self.to_tensor(img_tar)
        image, target = img_tar['img'], img_tar['tar']
        return image.detach(), target.unsqueeze(0).detach()
        # return image.detach(), target.detach()

    def __len__(self):
        return self.len


class TransformationTrain(object):

    def __init__(self, scales, cropsize):
        self.transforms = T.Compose([
            T.RandomResizedCrop(scales, cropsize),
            T.RandomHorizontalFlip(),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4
            ),
        ])

    def __call__(self, img_tar):
        img_tar = self.transforms(img_tar)
        return img_tar


class TransformationVal(object):

    def __call__(self, img_tar):
        img, tar = img_tar['img'], img_tar['tar']
        return dict(img=img, tar=tar)


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from lib.carla_cv2 import Carla
    from configs import cfg_factory
    cfg = cfg_factory['bisenetv2']
    # ds = CityScapes('./data/', mode='val')
    print("config: \n{}".format({key: value for key, value in cfg.__dict__.items()}))
    ds = Carla(cfg.train_img_root,
            cfg.train_img_anns,
            inputsize=cfg.input_size,
            mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = False,
                    num_workers = 4,
                    drop_last = True)
    for imgs, target in dl:
        print(len(imgs))
        for el in imgs:
            print("image.size(): {}".format(el.size()))
        for el in target:
            print("target.size(): {}".format(el.size()))
            print("target: {}".format(el))
            print("target.min(): {}".format(el.min()))
            print("target.max(): {}".format(el.max()))
        break
