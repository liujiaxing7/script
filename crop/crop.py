import argparse
import re

import numpy as np
from PIL import Image
import os
import cv2

target_size = (640, 400)
def process_image(image_path, crop_ratio):
    # 打开图像
    img = cv2.imread(image_path)

    # 计算中心裁剪的尺寸
    height, width, _ = img.shape
    crop_width = int(width * 0.9)
    crop_height = int(height * 0.9)

    # 中心裁剪
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    img_cropped = img[top:bottom, left:right]

    # 创建目标尺寸的图像
    target_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # 计算填充后的起始位置
    pad_x = (target_size[0] - crop_width) // 2
    pad_y = (target_size[1] - crop_height) // 2

    # 将裁剪后的图像粘贴到目标图像中心
    target_img[pad_y:pad_y+crop_height, pad_x:pad_x+crop_width] = img_cropped

    return target_img

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

def process_images(image_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的图像文件
    files = Walk(image_dir, ['jpg', 'png'])
    for filename in files:
        # if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = filename

            # 处理图像
            processed_image = process_image(image_path, 0.9)

            # 保存处理后的图像
            output_path = filename.replace("tof_scale", "tof_scale_out")
            if not os.path.exists(os.path.split(output_path)[0]):
                os.makedirs(os.path.split(output_path)[0])
            cv2.imwrite( output_path,processed_image)
            print(f"保存处理后的图像: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='image dir')
    parser.add_argument('--output_dir', type=str, help='image dir')
    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir)
