import argparse
import re

import cv2
from PIL import Image
import os

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

def resize_images_in_folder(input_path, output_path, scale):
    # 获取指定路径下所有图片文件
    file_list = Walk(input_path, ['jpg', 'jpeg', 'png', 'bmp'])

    # 遍历每一张图片，并将其resize成原来的2倍大小
    for img_file in file_list:
        # 打开图片文件
        img = cv2.imread(img_file)

        # 计算新的图片大小
        resized_image = img[::2, ::2]

        # 规定输出文件名
        output_filename = img_file.replace("data_2023_08_23_crop", "data_2023_08_23_crop_288_190")

        if not os.path.exists(os.path.split(output_filename)[0]):
            os.makedirs(os.path.split(output_filename)[0])

        # 保存resize后的图片
        cv2.imwrite(output_filename, resized_image)

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Image resizing')

# 添加命令行参数
parser.add_argument('--scale', type=float, default=2.0, help='scaling factor')
parser.add_argument('--input_path', type=str, default=None, help='scaling factor')
parser.add_argument('--output_path', type=str, default=None, help='scaling factor')

# 解析命令行参数
args = parser.parse_args()
resize_images_in_folder(args.input_path, args.output_path, args.scale)