#!/usr/bin/python
# -*- encoding: utf-8 -*-


import random
import math

import numpy as np
import cv2
import torch



class RandomResizedCrop(object):
    '''
    size should be a tuple of (H, W)
    '''
    def __init__(self, scales=(0.5, 1.), size=(384, 384)):
        self.scales = scales
        self.size = size

    def __call__(self, img_tar):
        if self.size is None:
            return img_tar

        img, tar = img_tar['img'], img_tar['tar']
        assert img.shape[:2] == tar.shape[:2]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        img_h, img_w = [math.ceil(el * scale) for el in img.shape[:2]]
        img = cv2.resize(img, (img_w, img_h))
        tar = cv2.resize(tar, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        if (img_h, img_w) == (crop_h, crop_w): return dict(img=img, tar=tar)
        pad_h, pad_w = 0, 0
        if img_h < crop_h:
            pad_h = (crop_h - img_h) // 2 + 1
        if img_w < crop_w:
            pad_w = (crop_w - img_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            tar = np.pad(tar, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        img_h, img_w, _ = img.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (img_h - crop_h)), int(sw * (img_w - crop_w))
        return dict(
            img=img[sh:sh+crop_h, sw:sw+crop_w, :].copy(),
            tar=tar[sh:sh+crop_h, sw:sw+crop_w].copy()
        )



class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_tar):
        if np.random.random() < self.p:
            return img_tar
        img, tar = img_tar['img'], img_tar['tar']
        assert img.shape[:2] == tar.shape[:2]
        return dict(
            img=img[:, ::-1, :],
            tar=tar[:, ::-1],
        )



class ColorJitter(object):

    def __init__(self, brightness=None, contrast=None, saturation=None):
        if not brightness is None and brightness >= 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast >= 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation >= 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, img_tar):
        img, tar = img_tar['img'], img_tar['tar']
        assert img.shape[:2] == tar.shape[:2]
        if not self.brightness is None:
            rate = np.random.uniform(*self.brightness)
            img = self.adj_brightness(img, rate)
        if not self.contrast is None:
            rate = np.random.uniform(*self.contrast)
            img = self.adj_contrast(img, rate)
        if not self.saturation is None:
            rate = np.random.uniform(*self.saturation)
            img = self.adj_saturation(img, rate)
        return dict(img=img, tar=tar,)

    def adj_saturation(self, img, rate):
        M = np.float32([
            [1+2*rate, 1-rate, 1-rate],
            [1-rate, 1+2*rate, 1-rate],
            [1-rate, 1-rate, 1+2*rate]
        ])
        shape = img.shape
        img = np.matmul(img.reshape(-1, 3), M).reshape(shape)/3
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def adj_brightness(self, img, rate):
        table = np.array([
            i * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[img]

    def adj_contrast(self, img, rate):
        table = np.array([
            74 + (i - 74) * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[img]




class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img_tar):
        img, tar = img_tar['img'], img_tar['tar']
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) # to rgb order
        img = torch.from_numpy(img).div_(255)
        dtype, device = img.dtype, img.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        img = img.sub_(mean).div_(std).clone()
        tar = torch.from_numpy(tar.astype(np.int64).copy()).clone()
        return dict(img=img, tar=tar)


class Compose(object):

    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, img_tar):
        for comp in self.do_list:
            img_tar = comp(img_tar)
        return img_tar




if __name__ == '__main__':
    #  from PIL import Image
    #  img = Image.open(imgpth)
    #  tar = Image.open(tarpth)
    #  print(tar.size)
    #  img.show()
    #  tar.show()
    import cv2
    img = cv2.imread(imgpth)
    tar = cv2.imread(tarpth, 0)
    tar = tar * 10

    trans = Compose([
        RandomHorizontalFlip(),
        RandomShear(p=0.5, rate=3),
        RandomRotate(p=0.5, degree=5),
        RandomScale([0.5, 0.7]),
        RandomCrop((768, 768)),
        RandomErasing(p=1, size=(36, 36)),
        ChannelShuffle(p=1),
        ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.5
        ),
        #  RandomEqualize(p=0.1),
    ])
    #  inten = dict(img=img, tar=tar)
    #  out = trans(inten)
    #  img = out['img']
    #  tar = out['tar']
    #  cv2.imshow('tar', tar)
    #  cv2.imshow('org', img)
    #  cv2.waitKey(0)


    ### try merge rotate and shear here
    img = cv2.imread(imgpth)
    tar = cv2.imread(tarpth, 0)
    img = cv2.resize(img, (1024, 512))
    tar = cv2.resize(tar, (1024, 512), interpolation=cv2.INTER_NEAREST)
    tar = tar * 10
    inten = dict(img=img, tar=tar)
    trans1 = Compose([
        RandomShear(p=1, rate=0.15),
        #  RandomRotate(p=1, degree=10),
    ])
    trans2 = Compose([
        #  RandomShearRotate(p_shear=1, p_rot=0, rate_shear=0.1, rot_degree=9),
        RandomHFlipShearRotate(p_flip=0.5, p_shear=1, p_rot=0, rate_shear=0.1, rot_degree=9),
    ])
    out1 = trans1(inten)
    img1 = out1['img']
    tar1 = out1['tar']
    #  cv2.imshow('tar', tar1)
    cv2.imshow('org1', img1)
    out2 = trans2(inten)
    img2 = out2['img']
    tar2 = out2['tar']
    #  cv2.imshow('tar', tar1)
    #  cv2.imshow('org2', img2)
    cv2.waitKey(0)
    print(np.sum(img1-img2))
    print('====')
    ####


    totensor = ToTensor(
        mean=(0.406, 0.456, 0.485),
        std=(0.225, 0.224, 0.229)
    )
    #  print(img[0, :2, :2])
    print(tar[:2, :2])
    out = totensor(out)
    img = out['img']
    tar = out['tar']
    print(img.size())
    #  print(img[0, :2, :2])
    #  print(tar[:2, :2])

    out = totensor(inten)
    img = out['img']
    print(img.size())
    print(img[0, 502:504, 766:768])

