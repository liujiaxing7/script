import os
import numpy as np
import copy
import codecs
import json
from glob import glob
import cv2
import shutil

# labelme_path = '/home/walt/Downloads/dataset/已标注-11/dev/shm/1672670153410605_bea9298c-3371-4f99-98f7-b271859acf11/1591039045110456321/'
labelme_path = '/mnt/sdb2/dataset/20230210/第1批/shm/1675929455170143_47bc5fa0-c901-450d-94ab-e7f5f1377bef/1623127369685729282/第一批桌子/cameradata-2/'
saved_path = '/mnt/sdb2/dataset/20230208/dev/xml'

from functools import reduce
import operator
import math


# 文件计数器
counter = 0


with codecs.open(labelme_path + "xml" + ".xml", "w", "utf-8") as xml:
    # 3.获取待处理文件
    files = glob(labelme_path + "*.json")
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
            xml.write('\t<image' + " file ='" + str(fileName) + ".jpg"+ "'>\n")
        else:
            xml.write('\t<image' + " file ='" + str(fileName)  + "'>\n")

        # cubePoints是一个数组，数组每一项是一个字典{x:  ,y:  }
        top, left, width, height = 0, 0, 0, 0
        if "cubePoints" in json_file:
            for pointInLists in json_file['cubePoints']:
                point_x_list = []
                point_y_list = []
                point_list = []
                '''
                int()函数是可以将字符串转换为整形，但是这个字符串如果是带小数得,就会转换报错
                '''
                for pointInList in pointInLists:
                    pointX = int(float(pointInList["x"]))
                    pointY = int(float(pointInList["y"]))
                    point_list.append([pointX, pointY])

                point_x_list = np.array(point_list)[:, 0].tolist()
                point_y_list = np.array(point_list)[:, 1].tolist()
                point_x_list2 = point_x_list.copy()
                point_y_list2 = point_y_list.copy()
                point_x_list.sort()
                point_y_list.sort()

                point_list = np.array(point_list)

                point_list_desk = point_list[point_list[:,1].argsort()][-4:]
                coords = point_list_desk.tolist()
                center = tuple(
                    map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
                coords_sorted = sorted(coords, key=lambda coord: (-135 - math.degrees(
                    math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=False)


                point_max_y = point_list_desk[point_list_desk[:,1].argsort()][0]
                point_max_y_index = coords_sorted.index(point_max_y.tolist())

                distance1 = max(coords_sorted[(point_max_y_index + 1)%len(coords_sorted)][0]-coords_sorted[(point_max_y_index + 2)%len(coords_sorted)][0], 0)
                distance2 = max(coords_sorted[(point_max_y_index + 2)%len(coords_sorted)][0]-coords_sorted[(point_max_y_index + 3)%len(coords_sorted)][0], 0)

                # if abs(distance1 - distance2)>35:
                if distance1 >= distance2:
                    point0 = coords_sorted[(point_max_y_index + 1)%len(coords_sorted)]
                else:
                    point0 = coords_sorted[(point_max_y_index + 2)%len(coords_sorted)]
                # else: point0 = coords_sorted[(point_max_y_index + 1)%len(coords_sorted)]



                point0_index = coords_sorted.index(point0)
                point1 = coords_sorted[(point0_index + 1)%len(coords_sorted)]
                point2 = coords_sorted[(point0_index + 2)%len(coords_sorted)]
                point3 = coords_sorted[(point0_index + 3)%len(coords_sorted)]

                point_desk_leg = [point0, point1, point2, point3]

                top = point_y_list[0]
                left = point_x_list[0]
                width = point_x_list[-1] - left
                height = point_y_list[-1] - top

                # <box top='100' left='5' width='387' height='296'>
                xml.write('\t\t<box' + " top='" + str(top) + "' left='" + str(left) + "' width='" + str(width) + "' height='" + str(height) + "'>\n")
                # <label>unlabelled</label>
                xml.write('\t\t\t<label>unlabelled</label>\n')

                # <part name='0' x='114' y='16'/>
                for index in range(len(point_desk_leg)):
                    xml.write('\t\t\t<part ' + "name='" + str(index) + "' x='" + str(point_desk_leg[index][0]) + "' y='" + str(point_desk_leg[index][1]) + "'/>\n")
                    index += 1

                xml.write('\t\t</box>\n')
            xml.write('\t</image>\n')
    xml.write('</images>\n')
    xml.write('</dataset>\n')

print("counter:", counter)