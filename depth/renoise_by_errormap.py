import argparse
import os
import re

import numpy as np
import cv2

# 读取 uint16 图像并转换为 NumPy 数组
# 47_1614045223060525.png
# 41_1614045697756412.png
# 30_1614045506542342.png

def Walk(path, suffix:list):
    file_list = []
    suffix = [s.lower() for s in suffix]
    if not os.path.exists(path):
        print("not exist path {}".format(path))
        return []

    if os.path.isfile(path):
        return [path,]

    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower()[1:] in suffix:
                file_list.append(os.path.join(root, file))

    file_list.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))

    return file_list

def mask(input_path, disp_path, output_path):
    file_list = Walk(input_path, ['jpg', 'jpeg', 'png', 'bmp'])
    file_list_disp = Walk(disp_path, ['jpg', 'jpeg', 'png', 'bmp'])

    for img_file , img_file_disp in zip(sorted(file_list), sorted(file_list_disp)):
        img_file_disp_save = output_path + img_file_disp.split("/", -4)[-1]
        image_ori = cv2.imread(img_file, -1)
        image_disp = cv2.imread(img_file_disp, -1)

        image = np.zeros_like(image_ori).astype("uint8")
        mask = image_ori > 20
        image[mask] = 255


        # 进行连通域分析
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        expanded_image = cv2.dilate(closed_image, kernel, iterations=1)

        mask = expanded_image==255

        image_disp[mask] = 0

        # 保存输出图像
        if not os.path.exists(os.path.split(img_file_disp_save)[0]):
            os.makedirs(os.path.split(img_file_disp_save)[0])
        cv2.imwrite(img_file_disp_save, image_disp)

parser = argparse.ArgumentParser(description='Image resizing')

# 添加命令行参数
parser.add_argument('--input_path', type=str, default=None, help='scaling factor', required=True)
parser.add_argument('--disp_path', type=str, default=None, help='scaling factor', required=True)
parser.add_argument('--output_path', type=str, default=None, help='scaling factor', required=True)

# 解析命令行参数
args = parser.parse_args()
mask(args.input_path, args.disp_path, args.output_path)