import argparse
import os

from cv2 import cv2
import numpy as np
from tqdm import tqdm


def contrast(path):
    img0=cv2.imread(path)
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)  # 彩色转为灰度图片
    m, n = img1.shape
    # 图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img1, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    rows_ext, cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1, rows_ext - 1):
        for j in range(1, cols_ext - 1):
            b += ((img1_ext[i, j] - img1_ext[i, j + 1]) ** 2 + (img1_ext[i, j] - img1_ext[i, j - 1]) ** 2 +
                  (img1_ext[i, j] - img1_ext[i + 1, j]) ** 2 + (img1_ext[i, j] - img1_ext[i - 1, j]) ** 2)

    cg = b / (4 * (m - 2) * (n - 2) + 3 * (2 * (m - 2) + 2 * (n - 2)) + 2 * 4)  # 对应上面48的计算公式
    print(cg)

def contrast1(path):
    img0=cv2.imread(path)
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)  # 彩色转为灰度图片
    print(img1.std())
    return img1.std()

def contrast2(path):
    img0=cv2.imread(path)
    Y = cv2.cvtColor(img0, cv2.COLOR_BGR2YUV)[:, :, 0]

    # compute min and max of Y
    min = np.min(Y)
    max = np.max(Y)

    # compute contrast
    contrast = (max - min) / (max + min)
    print(contrast)
    return contrast

if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_dir',type=str,default="/mnt/sdb1/Data/tiny_rgb_data/REMAP/TRAIN/batch7/images.txt")
    args=parser.parse_args()

    imgs_contrast=[]
    # for i in tqdm(sorted(os.listdir(args.input_dir))):
    with open(args.input_dir, 'r', encoding='utf-8') as f:
        for line in f:
            imgs_contrast.append(contrast1(line[:-1]))
    print('图像个数：{}，平均对比度：{}'.format(len(imgs_contrast),sum(imgs_contrast)/len(imgs_contrast)))