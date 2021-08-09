import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

file_path='/home/fandong/Code/Data/data6/detect_voc/VisDrone2019-DET-val'

classes_dict = {'1': 'ignored regions', '0': ['pedestrian', 'people'],
             '3': 'bicycle', '4': 'car', '5': 'van', '6': 'truck',
             '7': 'tricycle', '8': 'awning-tricycle', '9': 'bus',
             '10': 'motor', '11': 'others'}
def get_keys(d, value):
    for k, v in d.items():
        if type(v) is list and value in v:
            return k
        elif(v==value):
            return k

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(name):
    contain_person=False

    in_file = open(os.path.join(file_path,'xml/%s.xml' % (name)))
    out_path=os.path.join(file_path, 'labels/%s.txt' % (name))
    out_file = open(out_path, 'w')
    if not os.path.exists(file_path+'/labels'):
        os.mkdir(file_path+'/labels')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    # print(w,h)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        cls_list=get_keys(classes_dict,cls)
        if len(cls_list)==0 or int(cls_list[0])!=0:
            continue

        contain_person = True
        cls_id =cls_list[0]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)

        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    out_file.close()
    if os.path.getsize(out_path)==0:
        os.remove(out_path)
    return contain_person

print(os.path.join(file_path,'trainval.txt'))
image_ids = open(os.path.join(file_path,'trainval.txt')).read().strip().split()

list_file = open(os.path.join(file_path,'trainval.txt'), 'w')
for image_id in image_ids:
    is_contain_person=convert_annotation(image_id)
    if is_contain_person:
        list_file.write('images/%s.jpg\n' % (image_id))
list_file.close()

