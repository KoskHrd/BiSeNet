#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import logging
import argparse
import math
from tabulate import tabulate

from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist

from lib.models import model_factory
from configs import cfg_factory
from lib.logger import setup_logger
# from lib.cityscapes_cv2 import get_data_loader
# from lib.carla_cv2 import get_data_loader
from lib.carla_cv2 import get_data_loader, prepare_data_loader



class MscEvalV0(object):

    def __init__(self, scales=(0.5, ), flip=False, ignore_target=255):
        self.scales = scales
        self.flip = flip
        self.ignore_target = ignore_target

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        # if dist.is_initialized() and dist.get_rank() != 0:
        #     diter = enumerate(dl)
        # else:
        #     diter = enumerate(tqdm(dl))
        diter = enumerate(tqdm(dl, desc='eval'))
        for i, (imgs, target) in diter:
            # print("imgs.shape: {}".format(imgs.shape))
            # print("target.shape: {}".format(target.shape))
            N, _, H, W = target.shape  # 2, 1, 600, 800
            target = target.squeeze(1).cuda()
            size = target.size()[-2:]  # (600,800)
            probs = torch.zeros(
                    (N, n_classes, H, W), dtype=torch.float32).cuda().detach()
            for scale in self.scales:  # if scale==1.
                sH, sW = int(scale * H), int(scale * W)  # sH=600.,sW=800.
                img_sc = F.interpolate(imgs, size=(sH, sW),
                        mode='bilinear', align_corners=True)
                img_sc = img_sc.cuda()
                logits = net(img_sc)[0]  # torch.Size([2, 23, 600, 800])
                logits = F.interpolate(logits, size=size,
                        mode='bilinear', align_corners=True)
                probs += torch.softmax(logits, dim=1)
                if self.flip:
                    img_sc = torch.flip(img_sc, dims=(3, ))
                    logits = net(img_sc)[0]
                    logits = torch.flip(logits, dims=(3, ))
                    logits = F.interpolate(logits, size=size,
                            mode='bilinear', align_corners=True)
                    probs += torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)  # torch.Size([2, 600, 800])
            keep = target != self.ignore_target
            hist += torch.bincount(
                target[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)  # torch.Size([23, 23])
        # if dist.is_initialized():
        #     dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item(), ious.to('cpu').detach().numpy().copy()



class MscEvalCrop(object):

    def __init__(
        self,
        cropsize=1024,
        cropstride=2./3,
        flip=True,
        scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
        ignore_target=255,
    ):
        self.scales = scales
        self.ignore_target = ignore_target
        self.flip = flip
        # self.distributed = dist.is_initialized()

        self.cropsize = cropsize if isinstance(cropsize, (list, tuple)) else (cropsize, cropsize)
        self.cropstride = cropstride


    def pad_tensor(self, inten):
        N, C, H, W = inten.size()
        cropH, cropW = self.cropsize
        if cropH < H and cropW < W: return inten, [0, H, 0, W]
        padH, padW = max(cropH, H), max(cropW, W)
        outten = torch.zeros(N, C, padH, padW).cuda()
        outten.requires_grad_(False)
        marginH, marginW = padH - H, padW - W
        hst, hed = marginH // 2, marginH // 2 + H
        wst, wed = marginW // 2, marginW // 2 + W
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, net, crop):
        prob = net(crop)[0].softmax(dim=1)
        if self.flip:
            crop = torch.flip(crop, dims=(3,))
            prob += net(crop)[0].flip(dims=(3,)).softmax(dim=1)
            prob = torch.exp(prob)
        return prob


    def crop_eval(self, net, img, n_classes):
        cropH, cropW = self.cropsize
        stride_rate = self.cropstride
        img, indices = self.pad_tensor(img)
        N, C, H, W = img.size()

        strdH = math.ceil(cropH * stride_rate)
        strdW = math.ceil(cropW * stride_rate)
        n_h = math.ceil((H - cropH) / strdH) + 1
        n_w = math.ceil((W - cropW) / strdW) + 1
        prob = torch.zeros(N, n_classes, H, W).cuda()
        prob.requires_grad_(False)
        for i in range(n_h):
            for j in range(n_w):
                stH, stW = strdH * i, strdW * j
                endH, endW = min(H, stH + cropH), min(W, stW + cropW)
                stH, stW = endH - cropH, endW - cropW
                chip = img[:, :, stH:endH, stW:endW]
                prob[:, :, stH:endH, stW:endW] += self.eval_chip(net, chip)
        hst, hed, wst, wed = indices
        prob = prob[:, :, hst:hed, wst:wed]
        return prob


    def scale_crop_eval(self, net, img, scale, n_classes):
        N, C, H, W = img.size()
        new_hw = [int(H * scale), int(W * scale)]
        img = F.interpolate(img, new_hw, mode='bilinear', align_corners=True)
        prob = self.crop_eval(net, img, n_classes)
        prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
        return prob


    @torch.no_grad()
    def __call__(self, net, dl, n_classes):
        # dloader = dl if self.distributed and not dist.get_rank() == 0 else tqdm(dl)
        dloader = tqdm(dl)

        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        hist.requires_grad_(False)
        for i, (imgs, target) in enumerate(dloader):
            imgs = imgs.cuda()
            target = target.squeeze(1).cuda()
            N, H, W = target.shape
            probs = torch.zeros((N, n_classes, H, W)).cuda()
            probs.requires_grad_(False)
            for sc in self.scales:
                probs += self.scale_crop_eval(net, imgs, sc, n_classes)
            torch.cuda.empty_cache()
            preds = torch.argmax(probs, dim=1)

            keep = target != self.ignore_target
            hist += torch.bincount(
                target[keep] * n_classes + preds[keep],
                minlength=n_classes ** 2
                ).view(n_classes, n_classes)

        # if self.distributed:
        #     dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item(), ious.to('cpu').detach().numpy().copy()


@torch.no_grad()
def eval_model(net, cfg, gpu_count, img_root, img_anns, n_classes, anns_ignore):
    # is_dist = dist.is_initialized()
    # dl = get_data_loader(img_root, img_anns, imgs_per_gpu, None,
    #         None, mode='val', distributed=is_dist)
    # dl = get_data_loader(img_root, img_anns, imgs_per_gpu, None,
    #         anns_ignore, None, mode='val', distributed=False)
    dl = prepare_data_loader(img_root, img_anns, cfg.input_size, cfg.imgs_per_gpu, gpu_count,
            cfg.scales, cfg.cropsize, anns_ignore, mode='val', distributed=False)
    net.eval()

    heads, mious, eious = [], [], []
    logger = logging.getLogger()

    single_scale = MscEvalV0((1., ), False, ignore_target=anns_ignore)
    mIOU, eIOU = single_scale(net, dl, n_classes)
    heads.append('single_scale')
    mious.append(mIOU)
    logger.info('single mIOU is: %s', mIOU)
    eious.append(eIOU)
    logger.info('single eachIOU is: \n{}'.format(eIOU))

    single_crop = MscEvalCrop(
        # cropsize=1024,
        cropsize=800,
        cropstride=2. / 3,
        flip=False,
        scales=(1., ),
        # ignore_target=255,
        ignore_target=anns_ignore,
    )
    mIOU, eIOU = single_crop(net, dl, n_classes)
    heads.append('single_scale_crop')
    mious.append(mIOU)
    logger.info('single scale crop mIOU is: %s', mIOU)
    eious.append(eIOU)
    logger.info('single scale crop eachIOU is: \n{}'.format(eIOU))

    # ms_flip = MscEvalV0((0.5, 0.75, 1, 1.25, 1.5, 1.75), True, ignore_target=anns_ignore)
    ms_flip = MscEvalV0((0.75, 1.0, 1.25, 1.5, 1.75, 2.0), True, ignore_target=anns_ignore)
    mIOU, eIOU = ms_flip(net, dl, n_classes)
    heads.append('ms_flip')
    mious.append(mIOU)
    logger.info('ms flip mIOU is: %s', mIOU)
    eious.append(eIOU)
    logger.info('ms flip eachIOU is: \n{}'.format(eIOU))

    ms_flip_crop = MscEvalCrop(
        # cropsize=1024,
        cropsize=800,
        cropstride=2. / 3,
        flip=True,
        # scales=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
        scales=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
        ignore_target=anns_ignore,
    )
    mIOU, eIOU = ms_flip_crop(net, dl, n_classes)
    heads.append('ms_flip_crop')
    mious.append(mIOU)
    logger.info('ms crop mIOU is: %s', mIOU)
    eious.append(eIOU)
    logger.info('ms crop eachIOU is: \n{}'.format(eIOU))
    return heads, mious, eious


def evaluate(cfg, weight_pth):
    logger = logging.getLogger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_count = torch.cuda.device_count() if device.type == 'cuda' else None

    ## model
    logger.info('setup and restore model')
    net = model_factory[cfg.model_type](cfg.n_classes)
    #  net = BiSeNetV2(cfg.n_classes)
    net.load_state_dict(torch.load(weight_pth))
    # net.cuda()
    net.to(device)

    # is_dist = dist.is_initialized()
    # if is_dist:
    #     local_rank = dist.get_rank()
    #     net = nn.parallel.DistributedDataParallel(
    #         net,
    #         device_ids=[local_rank, ],
    #         output_device=local_rank
    #     )

    ## evaluator
    # heads, mious, eious = eval_model(net, 2, device_count, cfg.val_img_root, cfg.val_img_anns,
    #     cfg.n_classes, cfg.anns_ignore)
    heads, mious, eious = eval_model(net, cfg, device_count, cfg.val_img_root, cfg.val_img_anns,
        cfg.n_classes, cfg.anns_ignore)
    # logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
    logger.info('\n' + tabulate([mious, ], headers=heads, tablefmt='github', floatfmt=".8f"))
    logger.info('\n' + tabulate(eious.transpose(), headers=heads, tablefmt='github', floatfmt=".8f"))


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank',
                    type=int, default=-1,)
    # parse.add_argument('--weight-path', dest='weight_pth', type=str,
    #                    default='./res/model_final.pth',)
    # parse.add_argument('--weight-path', dest='weight_pth', type=str,
    #                 default='./res/weight/model_final_bisenetv2-train-info-2021-04-25-22-37-00.pth',)
    parse.add_argument('--weight-path', dest='weight_path', type=str,
                    default='./res/weight/model_best_valid_loss.pth',)
    parse.add_argument('--port', dest='port', type=int, default=44553,)
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
    return parse.parse_args()


def main():
    np.set_printoptions(precision=8, floatmode='maxprec', suppress=True)
    args = parse_args()
    cfg = cfg_factory[args.model]
    if not args.local_rank == -1:
        torch.cuda.set_device(args.local_rank)
        # dist.init_process_group(backend='nccl',
        # init_method='tcp://127.0.0.1:{}'.format(args.port),
        # world_size=torch.cuda.device_count(),
        # rank=args.local_rank
        # )
    if not os.path.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('Trial-{}-eval-{}'.format(cfg.model_type,cfg.log_level), cfg.respth, cfg.log_level)
    evaluate(cfg, args.weight_path)


if __name__ == "__main__":
    main()
