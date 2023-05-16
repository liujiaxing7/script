import shutil
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
import os
from pathlib import Path
from xml.etree import ElementTree  # 导入ElementTree模块

import cv2

# from get_dir_image import conver_list_dict, readDirImg
import argparse

def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def apped_obj(obj,obj_name,x,y,w,h):
    name = Element("name")
    name.text = obj_name
    pose = Element("pose")
    pose.text = "Unspecified"
    truncated = Element("truncated")
    truncated.text = "0"
    difficult = Element("difficult")
    difficult.text = "0"
    distance = Element("distance")
    distance.text = "0"
    obj.append(name)
    obj.append(pose)
    obj.append(truncated)
    obj.append(difficult)
    obj.append(distance)

    bndbox = ET.Element('bndbox')
    obj.append(bndbox)
    xmin = ET.Element('xmin')
    xmin.text = str(x)
    bndbox.append(xmin)
    ymin = ET.Element('ymin')
    ymin.text = str(y)
    bndbox.append(ymin)
    xmax = ET.Element('xmax')
    xmax.text = str(x + w)
    bndbox.append(xmax)
    ymax = ET.Element('ymax')
    ymax.text = str(y + h)
    bndbox.append(ymax)

def read_xml_list(file):
    file_list = []
    with open(file, 'r') as file_r:
        lines = file_r.readlines()
        for line in lines:
            value = line.strip()
            file_list.append(value)
    return file_list

def imglab2lableimg(imglab_xml):
    # print(imglab_xml)
    # doc = ET.parse(imglab_xml)
    # root = doc.getroot()
    # print(root)

    images = read_xml_list(imglab_xml)

    # images = root.find('images')
    # print(images)

    # image = images.findall('image')
    for file in images:

        img_path_save = file.replace("JPEGImages", "JPEGImages_distort").replace("labels", "labels_distort").replace(".jpg", "_distort.jpg").replace(".txt", "_distort.txt")

        if not os.path.exists(os.path.split(img_path_save)[0]):
            os.makedirs(os.path.split(img_path_save)[0])

        shutil.copy(file, img_path_save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, help='image dir')
    parser.add_argument('--imglab-xml', type=str, help='xml from imglab')
    opt = parser.parse_args()
    image_root_dir = opt.image_dir
    # imglab_xml = opt.imglab_xml
    # image_list = read_xml_list(image_root_dir)
    # # print(image_list)
    # image_dict = conver_list_dict(image_list)
    # print(image_dict)
    imglab2lableimg(image_root_dir)
