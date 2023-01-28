import os
import re

import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil

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

labelme_path = '/mnt/sdb2/dataset/20230119/桌子'
saved_path = '/mnt/sdb2/dataset/20230119/桌子/xml'
# 3.获取待处理文件
# files = glob(labelme_path + "*.json")
files = Walk(labelme_path, ['json'])
print(files)
# 4.读取标注信息并写入 xml
for json_file_ in files:
    print(json_file_)
    # json_filename = labelme_path + json_file_ + ".json"
    # print(json_filename)
    json_file = json.load(open(json_file_, "r", encoding="utf-8"))
    # height, width, channels = cv2.imread(labelme_path + json_file_ + ".jpg").shape
    height, width, channels = json_file["imageHeight"], json_file["imageWidth"], 1

    output_file = labelme_path + "xml/" + json_file_.replace(".json", "").replace(labelme_path, "") + ".xml"
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with codecs.open(labelme_path + "xml/" + json_file_.replace(".json", "").replace(labelme_path, "") + ".xml", "w", "utf-8") as xml:

        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'CELL_data' + '</folder>\n')
        xml.write('\t<filename>' + json_file_.replace(labelme_path, "").replace(".json", "") + ".jpg" + '</filename>\n')

        xml.write('\t<source>\n')
        xml.write('\t\t<database>CELL Data</database>\n')
        # xml.write('\t\t<annotation>CELL</annotation>\n')
        # xml.write('\t\t<image>bloodcell</image>\n')
        # xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')

        # xml.write('\t<owner>\n')
        # xml.write('\t\t<flickrid>NULL</flickrid>\n')
        # xml.write('\t\t<name>CELL</name>\n')
        # xml.write('\t</owner>\n')

        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')

        xml.write('\t\t<segmented>0</segmented>\n')# 是否用于分割（在图像物体识别中01无所谓）

        # cName = json_file["categories"]
        # cName = json_file["label"]
        # Name = cName[0]["name"]
        # print(Name)
        # print(json_file["version"])
        for multi in json_file["shapes"]:
            # points = np.array(multi["bbox"])
            points = np.array(multi["points"])
            Name = multi['label']
            labelName = Name
            # xmin = points[0]
            # xmax = points[0]+points[2]
            # ymin = points[1]
            # ymax = points[1]+points[3]
            xmin = points[0][0]
            xmax = points[1][0]
            ymin = points[0][1]
            ymax = points[1][1]
            label = Name
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + labelName + '</name>\n')# 物体类别
                xml.write('\t\t<pose>Unspecified</pose>\n')# 拍摄角度
                xml.write('\t\t<truncated>0</truncated>\n')# 是否被截断（0表示完整）
                xml.write('\t\t<difficult>0</difficult>\n')# 目标是否难以识别（0表示容易识别）
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(int(xmin)) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(int(ymin)) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(int(xmax)) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(int(ymax)) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                # print(json_filename, xmin, ymin, xmax, ymax, label)
                # print( xmin, ymin, xmax, ymax, label)
        xml.write('</annotation>')