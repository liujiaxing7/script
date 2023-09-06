import argparse

import cv2
import numpy as np


def read_xml_list(file):
    file_list = []
    with open(file, 'r') as file_r:
        lines = file_r.readlines()
        for line in lines:
            value = line.strip()
            file_list.append(value)
    return file_list

def select(file):
    depth_list = read_xml_list(file)
    all_depth_mask = {}
    for depth_file in depth_list:
        # print(depth_file)
        depth_img = cv2.imread(depth_file, -1)
        depth_img = depth_img/100
        depth_img_mask = depth_img < 350
        depth_img_mask_num = np.sum(depth_img_mask)
        all_depth_mask[depth_file] = depth_img_mask_num
    sorted_dict = dict(sorted(all_depth_mask.items(), key=lambda item: item[1], reverse=True))
    output_dict = dict(list(sorted_dict.items())[0:5000])

    with open("output.txt", "w") as file:
        for key in output_dict.keys():
            print(key)
            file.write(key + "\n")


parser = argparse.ArgumentParser(description='Image resizing')

# 添加命令行参数
parser.add_argument('--file', type=str, default=None, help='scaling factor')
args = parser.parse_args()
select(args.file)