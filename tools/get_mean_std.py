#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import cv2
import numpy as np
from tqdm import tqdm
import glob


def calculate_mean_std(path):
    # folder = os.listdir(path)
    folder = glob.glob(path, recursive=False)

    mean = []
    std = []

    R_mean = 0.0
    G_mean = 0.0
    B_mean = 0.0
    # for img_name in tqdm(folder, desc="calculate_mean_std"):
        # image = cv2.imread(os.path.join(path+img_name))/255.
    for img_name in tqdm(folder, desc="calculate_mean_std"):
        image = cv2.imread(img_name)/255.
        R_mean += np.mean(image[:,:,2])
        G_mean += np.mean(image[:,:,1])
        B_mean += np.mean(image[:,:,0])
    R_mean = R_mean / len(folder)
    G_mean = G_mean / len(folder)
    B_mean = B_mean / len(folder)
    mean.extend([R_mean, G_mean, B_mean])
    #print(mean)

    R_std = 0.0
    G_std = 0.0
    B_std = 0.0
    # for img_name in tqdm(folder, desc="calculate_mean_std"):
        # image = cv2.imread(os.path.join(path+img_name))/255.
    for img_name in tqdm(folder, desc="calculate_mean_std"):
        image = cv2.imread(img_name)/255.
        image_size = image.shape[0]*image.shape[1]
        R_std += np.sum(np.power(image[:,:,2] - R_mean, 2)) / image_size
        G_std += np.sum(np.power(image[:,:,1] - G_mean, 2)) / image_size
        B_std += np.sum(np.power(image[:,:,0] - B_mean, 2)) / image_size
    R_std = np.sqrt(R_std / len(folder))
    G_std = np.sqrt(G_std / len(folder))
    B_std = np.sqrt(B_std / len(folder))
    std.extend([R_std, G_std, B_std])
    #print(std)
    return mean, std

if __name__ == "__main__":
    image_path = "./dataset/carla/data*/CameraRGB/*.png" # glob.iglob(img_path)>>dataABC,dataD
    mean, std = calculate_mean_std(path=image_path)
    print('mean={}\nstd={}\n#value=[R,G,B]'.format(mean,std))

    # mean=[0.35245398366625774, 0.3581336873774504, 0.35212403845792456]
    # std=[0.26050246625763057, 0.2570952503100638, 0.2654258674345489]
    # value=[R,G,B]

