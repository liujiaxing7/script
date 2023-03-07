#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: imageprocess.py
@time: 2021/3/29 下午3:16
@desc: 
'''
import cv2
import numpy as np
import os

W, H = 640, 400


def GetFishEye(config, flag='l', scale=False):
    r = config.getNode('R' + flag).mat()
    p = config.getNode('P' + flag).mat()
    k = config.getNode('K' + flag).mat()
    if scale:
        p[p > 1] /= 2
        k[k > 1] /= 2
    d = config.getNode('D' + flag).mat()

    fisheye_x, fisheye_y = np.ndarray((H, W), np.float32), np.ndarray((H, W), np.float32)

    cv2.fisheye.initUndistortRectifyMap(k, d, r, p[0:3, 0:3], (W, H), cv2.CV_32FC1, fisheye_x, fisheye_y)
    return fisheye_x, fisheye_y


def convert(size, box):
    '''
    Invoked by **voc2yolo**

    :param size:
    :param box:
    :return: 4 values in a tuple representing the bbox in the format of (x, y, w, h)
    '''
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    if w >= 1:
        w = 0.99
    if h >= 1:
        h = 0.99
    return [x, y, w, h]


def GetDarknetLabels(img, ann_file, fisheye_x, fisheye_y, flip):
    if not os.path.exists(ann_file):
        # raise IOError("no such file: ", ann_file)
        return

    with open(ann_file, 'r') as f:
        ann_file_lines = f.readlines()
        box_label = []
    for ann_file_line in ann_file_lines:
        line_temp = ann_file_line.strip().split(' ')
        box_label.append(list(map(float, line_temp)))

    width = np.array(img).shape[1]
    height = np.array(img).shape[0]

    dst_path = ann_file.replace('labels', 'labels_distort')
    base_path, basename = os.path.split(dst_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    txtname = dst_path[:-4] + '.txt'
    txtfile = os.path.join(dst_path, txtname)

    points_distort_all = []
    have_out = True
    for i in box_label:
        box_distort = []
        xmin = i[1] * width - i[3] * width / 2.0
        xmax = xmin + i[3] * width
        ymin = i[2] * height - i[4] * height / 2.0
        ymax = ymin + i[4] * height

        if xmin<5 or ymin<5 or xmax>634 or ymax>394:
            # print(ann_file)
            # print("labels:", i[0])
            have_out = False
            continue


        xmin = np.clip(xmin, 0, 639)
        xmax = np.clip(xmax, 0, 639)
        ymin = np.clip(ymin, 0, 399)
        ymax = np.clip(ymax, 0, 399)

        if flip:
            ymax_flip = height - ymin - 1
            ymin_flip = height - ymax - 1
            xmax_flip = width - xmin - 1
            xmin_flip = width - xmax - 1
            ymax = ymax_flip
            ymin = ymin_flip
            xmax = xmax_flip
            xmin = xmin_flip

        xmed = (xmax + xmin) / 2.0
        ymed = (ymax + ymin) / 2.0

        xmin_dis = fisheye_x[int(ymin), int(xmin)]
        xmax_dis = fisheye_x[int(ymax), int(xmax)]
        xmed_top_x = fisheye_x[int(ymin), int(xmed)]
        xmed_right_x = fisheye_x[int(ymed), int(xmax)]
        xmed_bottom_x = fisheye_x[int(ymax), int(xmed)]
        xmed_left_x = fisheye_x[int(ymed), int(xmin)]

        ymin_dis = fisheye_y[int(ymin), int(xmin)]
        ymax_dis = fisheye_y[int(ymax), int(xmax)]
        ymed_top_y = fisheye_y[int(ymin), int(xmed)]
        ymed_right_y = fisheye_y[int(ymed), int(xmax)]
        ymed_bottom_y = fisheye_y[int(ymax), int(xmed)]
        ymed_left_y = fisheye_y[int(ymed), int(xmin)]

        xminest = min(xmin_dis, xmax_dis, xmed_top_x, xmed_bottom_x, xmed_left_x, xmed_right_x)
        xmaxest = max(xmin_dis, xmax_dis, xmed_top_x, xmed_bottom_x, xmed_left_x, xmed_right_x)
        yminest = min(ymin_dis, ymax_dis, ymed_top_y, ymed_bottom_y, ymed_left_y, ymed_right_y)
        ymaxest = max(ymin_dis, ymax_dis, ymed_top_y, ymed_bottom_y, ymed_left_y, ymed_right_y)

        if xminest < 1:
            print(ann_file + "  :  " + str(xminest))

        if flip:
            ymaxest_flip = height - yminest - 1
            yminest_flip = height - ymaxest - 1
            xmaxest_flip = width - xminest - 1
            xminest_flip = width - xmaxest - 1
            ymaxest = ymaxest_flip
            yminest = yminest_flip
            xmaxest = xmaxest_flip
            xminest = xminest_flip

        x_distort_center = ((xmaxest + xminest) / 2.0) / width
        y_distort_center = ((ymaxest + yminest) / 2.0) / height
        w_distort = (xmaxest - xminest) / width
        h_distort = (ymaxest - yminest) / height

        box_distort.extend([int(i[0]), x_distort_center, y_distort_center, w_distort, h_distort])
        if not len(i) == 5:
            points_distort = []
            for j in range(4):
                point_distort_x = fisheye_x[int(i[j * 2 + 6] * height), np.clip(int(i[j * 2 + 5] * width), 0,
                                                                                639)] / width
                point_distort_y = fisheye_y[int(i[j * 2 + 6] * height), np.clip(int(i[j * 2 + 5] * width), 0,
                                                                                639)] / height
                points_distort.extend([point_distort_x, point_distort_y])
            box_distort.extend(points_distort)
        points_distort_all.append(box_distort)
    with open(txtfile, "w+", encoding='UTF-8') as out_file:
        if have_out:
            for box_distort_i in points_distort_all:
                out_file.write(" ".join([str(a) for a in box_distort_i]) + '\n')
        # print("xmin:", fisheye_x[182, 284])
        # print("xmax:", fisheye_x[286, 389])
        # print("ymin:", fisheye_y[182, 284])
        # print("ymax:", fisheye_y[286, 389])

    return 0


def ReadPara(file, scale):
    if not os.path.exists(file):
        print("not exist file <", file, ">.")
        exit(0)

    config = cv2.FileStorage(file, cv2.FILE_STORAGE_READ)
    fisheye_x_l, fisheye_y_l = GetFishEye(config, 'l', scale)
    fisheye_x_r, fisheye_y_r = GetFishEye(config, 'r', scale)

    config.release()

    return fisheye_x_l, fisheye_y_l, fisheye_x_r, fisheye_y_r


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'JPEGImages' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]


def Remap(image, fisheye_x, fisheye_y, path, flip):
    labels_file = img2label_paths([path])
    boxlabels = GetDarknetLabels(image, labels_file[0], fisheye_x, fisheye_y, flip)
