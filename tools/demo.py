import sys
sys.path.insert(0, '.')
import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

from lib.models import model_factory
from configs import cfg_factory


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

def vis_save(pred, dstpth):
    height,width = pred.shape
    pred_dst = np.empty((height,width,3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # pred_dst[i,j,:] = lookup_table[pred_gray[i,j]]
            pred_dst[i,j,:] = lookup_table[pred[i,j]]
    pred_dst = Image.fromarray(pred_dst)
    pred_dst.save(dstpth)

torch.set_grad_enabled(False)
np.random.seed(123)

if __name__ == "__main__":
    # args
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
    # parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
    parse.add_argument('--weight-path', type=str,
        default='./res/weight/model_bestValidLoss-bisenetv2-train-info-2021-05-10-12-53-29.pth',)
    # parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
    parse.add_argument('--img-path', dest='img_path', type=str,
        default='./dataset/carla/dataE/CameraRGB/02_00_008.png',)
    args = parse.parse_args()
    cfg = cfg_factory[args.model]


    # palette and mean/std
    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
    # mean = torch.tensor([0.3257, 0.3690, 0.3223], dtype=torch.float32).view(-1, 1, 1)
    # std = torch.tensor([0.2112, 0.2148, 0.2115], dtype=torch.float32).view(-1, 1, 1)
    mean = torch.tensor([0.3524, 0.3581, 0.3521], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor([0.2605, 0.2571, 0.2654], dtype=torch.float32).view(-1, 1, 1)

    # define model
    net = model_factory[cfg.model_type](cfg.n_classes)
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    net.eval()
    net.cuda()

    # prepare data
    img = cv2.imread(args.img_path)
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    img = torch.from_numpy(img).div_(255).sub_(mean).div_(std).unsqueeze(0).cuda()

    # inference
    out = net(img)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
    # pred = palette[out]
    # cv2.imwrite('./res/pred-02_00_008.jpg', pred)
    vis_save(out, os.path.join(cfg.respth, 'pred-02_00_008-colored.jpg'))