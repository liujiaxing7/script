import os
import numpy as np
import copy
import codecs
import json
from glob import glob
import cv2
import shutil

# labelme_path = '/home/walt/Downloads/dataset/已标注-11/dev/shm/1672670153410605_bea9298c-3371-4f99-98f7-b271859acf11/1591039045110456321/'
labelme_path = '/mnt/sdb2/dataset/20230119/桌子/'
saved_path = '/mnt/sdb2/dataset/20230119/桌子/xml/'


# 文件计数器
counter = 0


json_files = glob(labelme_path + "*.json")
img_files = glob(labelme_path + "*.jpg")
for json_file_ in json_files:
    if "-" in json_file_:
        json_file = json.load(open(json_file_, "r", encoding="utf-8"))
        json_file_name = json_file_.split(' ')[0]
        json_file_name += '.json'
        json_file_name = json_file_name.replace("桌子","桌子_")

        save = json.dumps(json_file)
        f2 = open(json_file_name, 'w')
        f2.write(save)
        f2.close()

for img_file_ in img_files:
    if "-" in img_file_:
        img_file_name = img_file_.split(' ')[0]
        img_file_name += '.jpg'
        img_file_name = img_file_name.replace("桌子","桌子_")

        save = cv2.imread(img_file_)
        cv2.imwrite(img_file_name,save)