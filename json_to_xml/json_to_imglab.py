import os
import re

import numpy as np
import copy
import codecs
import json
from glob import glob
import cv2
import shutil

# labelme_path = '/home/walt/Downloads/dataset/已标注-11/dev/shm/1672670153410605_bea9298c-3371-4f99-98f7-b271859acf11/1591039045110456321/'
labelme_path = '/mnt/sdb2/dataset/20230212/第11批/'
saved_path = '/mnt/sdb2/dataset/20230212/第11批/xml/'


# 文件计数器
counter = 0

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

with codecs.open(labelme_path + "xml" + ".xml", "w", "utf-8") as xml:
    # 3.获取待处理文件
    files = Walk(labelme_path, ['json'])
    # files = glob(labelme_path + "*.json")
    print(files)
    # 4.读取标注信息并写入 xml
    xml.write('<dataset>\n')
    xml.write('<images>\n')
    for json_file_ in files:
        # json_filename = labelme_path + json_file_ + ".json"
        # print(json_filename)
        json_file = json.load(open(json_file_, "r", encoding="utf-8"))

        # 获取文件名
        fileName = json_file["image_name"]
        if "-" in fileName:
            fileName = fileName.split(' ')[0]

        xml.write('\t<image' + " file ='" + str(fileName) + "'>\n")

        # cubePoints是一个数组，数组每一项是一个字典{x:  ,y:  }

        top, left, width, height = 0, 0, 0, 0
        # if "cubePoints" in json_file:

        #     for pointInList in json_file['cubePoints']:
        if "cubePoints" in json_file:
            for pointInLists in json_file['cubePoints']:
                point_x_list = []
                point_y_list = []
                '''
                int()函数是可以将字符串转换为整形，但是这个字符串如果是带小数得,就会转换报错
                '''
                for pointInList in pointInLists:
                    '''
                    int()函数是可以将字符串转换为整形，但是这个字符串如果是带小数得,就会转换报错
                    '''
                    pointX = int(float(pointInList["x"]))
                    pointY = int(float(pointInList["y"]))
                    point_x_list.append(pointX)
                    point_y_list.append(pointY)
            # np.argsort(point_x_list)
            # np.argsort(point_y_list)
            # 不能使用= 进行直接复制，复制的是list的地址，这两个列表是完全等价的，修改其中任何一个列表都会影响到另一个列表。
            # 使用copy进行浅拷贝（因为只有一层，无需深拷贝）
                point_x_list2 = point_x_list.copy()
                point_y_list2 = point_y_list.copy()
                point_x_list.sort()
                point_y_list.sort()
                top = point_y_list[0]
                left = point_x_list[0]
                width = point_x_list[-1] - left
                height = point_y_list[-1] - top

                # <box top='100' left='5' width='387' height='296'>
                xml.write('\t\t<box' + " top='" + str(top) + "' left='" + str(left) + "' width='" + str(width) + "' height='" + str(height) + "'>\n")
                # <label>unlabelled</label>
                xml.write('\t\t\t<label>unlabelled</label>\n')

                # <part name='0' x='114' y='16'/>
                for index in range(len(point_x_list)):
                    xml.write('\t\t\t<part ' + "name='" + str(index) + "' x='" + str(point_x_list2[index]) + "' y='" + str(point_y_list2[index]) + "'/>\n")
                    index += 1

                xml.write('\t\t</box>\n')
            xml.write('\t</image>\n')
    xml.write('</images>\n')
    xml.write('</dataset>\n')

print("counter:", counter)