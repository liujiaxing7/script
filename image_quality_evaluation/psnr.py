import argparse

import cv2
import numpy as np
import math


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def Laplacian(path):
    img=cv2.imread(path,0)
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    return cv2.Laplacian(img,cv2.CV_64F).var()

if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_dir',type=str,default="/mnt/sdb1/Data/tiny_rgb_data/REMAP/TRAIN/batch7/images.txt")
    args=parser.parse_args()

    imgs_clarity=[]
    # for i in tqdm(sorted(os.listdir(args.input_dir))):
    with open(args.input_dir, 'r', encoding='utf-8') as f:
        for line in f:
            imgs_clarity.append(Laplacian(line[:-1]))
    print('图像个数：{}，平均清晰度：{}'.format(len(imgs_clarity),sum(imgs_clarity)/len(imgs_clarity)))