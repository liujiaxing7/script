#!/usr/bin/python3 python
# encoding: utf-8

import sys, os
import cv2
import numpy as np
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
from utils.file import MkdirSimple, Walk

CONFIG_FILE = 'config.yaml'

def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="image file or dir")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir")
    parser.add_argument("--flip", type=bool, default=False, help="flip up-down")
    parser.add_argument("--module", type=bool, default=False, help="data capture module")

    args = parser.parse_args()
    return args

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'JPEGImages' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]

def GetDarknetLabels(img, ann_file, img_path):
    if not os.path.exists(ann_file):
        # raise IOError("no such file: ", ann_file)
        print(img_path)
        return

    with open(ann_file, 'r') as f:
        ann_file_lines = f.readlines()
        box_label = []
    for ann_file_line in ann_file_lines:
        line_temp = ann_file_line.strip().split(' ')
        box_label.append(list(map(float, line_temp)))

    if len(box_label) == 0:
        print("empty:", img_path)

    width = np.array(img).shape[1]
    height = np.array(img).shape[0]

    for i in box_label:

        if not i[0] == 7:
            continue

        xmin = i[1] * width - i[3] * width / 2.0
        xmax = xmin + i[3] * width
        ymin = i[2] * height - i[4] * height / 2.0
        ymax = ymin + i[4] * height

        xmin = int(np.clip(xmin, 0, 639))
        xmax = int(np.clip(xmax, 0, 639))
        ymin = int(np.clip(ymin, 0, 399))
        ymax = int(np.clip(ymax, 0, 399))

        #外扩截取:
        xmin = np.clip(xmin, 0, 639)
        ymin = np.clip(ymin, 0, 639)
        xmax = np.clip(xmax, 0, 639)
        ymax = np.clip(ymax, 0, 639)


        crop_img = img[ymin:ymax, xmin:xmax]
        save_path = img_path.replace("JPEGImages", "JPEGImages_scale_out")
        ann_file_savepath = ann_file.replace("labels", "labels_scale_out")

        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])

        if not os.path.exists(os.path.split(ann_file_savepath)[0]):
            os.makedirs(os.path.split(ann_file_savepath)[0])

        point_new = []

        if len(i)>5:
            for p in range(4):
                point_x_new = i[p * 2 + 5] * width - xmin
                point_y_new = i[p * 2 + 6] * height - ymin
                point_new.extend([point_x_new/(xmax - xmin), point_y_new/(ymax - ymin)])

            with open(ann_file_savepath, "w+", encoding='UTF-8') as out_file:
                    out_file.write(" ".join([str(a) for a in point_new]) + '\n')

        try:
            cv2.imwrite(save_path, crop_img)
        except:
            print(img_path)
            continue

    return 0

def main():
    args = GetArgs()

    root = len(os.path.dirname(args.input.rstrip("/")))
    dirs = os.listdir(args.input)
    for d in dirs:
        if 'rgb' in d:
            continue
        d = os.path.join(args.input, d)
        if not os.path.isdir(d):
            continue
        print("in dir: ", d)
        files = Walk(d, ['jpg', 'png'])

        for f in tqdm(files):
            image = cv2.imread(f)
            labels_file = img2label_paths([f])
            boxlabels = GetDarknetLabels(image, labels_file[0], f)


if __name__ == '__main__':
    main()