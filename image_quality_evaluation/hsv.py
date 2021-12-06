import argparse

import numpy as np
from cv2 import cv2


def rgb2hsv(img_path, arr=None):

    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv=hsv[..., 2]
    cv2.imwrite('brightness.png',hsv)

    return hsv[hsv<100].mean()

if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_dir',type=str,default="/mnt/sdb1/Data/tiny_rgb_data/REMAP/TRAIN/batch7/images.txt")
    args=parser.parse_args()

    imgs_hsv=[]
    # for i in tqdm(sorted(os.listdir(args.input_dir))):
    with open(args.input_dir, 'r', encoding='utf-8') as f:
        for line in f:
            imgs_hsv.append(rgb2hsv(line[:-1]))
    print('图像个数：{}，平均亮度：{}'.format(len(imgs_hsv),sum(imgs_hsv)/len(imgs_hsv)))