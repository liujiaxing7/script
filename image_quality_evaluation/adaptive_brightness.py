import argparse

import numpy as np
import cv2


def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def aug(path):
    src=cv2.imread(path)
    """图像亮度增强"""
    if get_lightness(src) > 130:
        print("图片亮度足够，不做增强")
        return False

    print("这里有一张暗图")
    cv2.imwrite('old.png', src)
    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。

    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

    return out

def aug1(path):
    src=cv2.imread(path)
    """图像亮度增强"""
    if get_lightness(src) > 130:
        print("图片亮度足够，不做增强")
        return False
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    hsv_image[..., 2]=cv2.equalizeHist(hsv_image[..., 2])
    out=cv2.cvtColor(hsv_image,cv2.COLOR_HSV2RGB)

    return out


def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness

if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_dir',type=str,default="/mnt/sdb1/Data/data7/record_b2_remap/record_b2/record_v2/images.txt")
    args=parser.parse_args()

    imgs_hsv=[]
    # for i in tqdm(sorted(os.listdir(args.input_dir))):
    with open(args.input_dir, 'r', encoding='utf-8') as f:
        for line in f:
            img=aug(line[:-1])
            if type(img)==bool:
                continue
            else:
                cv2.imwrite('new.png', img)
    # print('图像个数：{}，平均亮度：{}'.format(len(imgs_hsv),sum(imgs_hsv)/len(imgs_hsv)))


