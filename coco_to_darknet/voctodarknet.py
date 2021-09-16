import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

# 20190227@new-only 2007 data
# sets=[('2007', 'train'), ('2007', 'val'), ('2007_test', 'test')]
sets = ['val']
# classes = ['1', '2', '3','4','5','6','7','8','9','10','11', '12', '13','14','15','16','17','18','19','20']
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']


def convert(size, box):
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
    return (x, y, w, h)


def convert_annotation(image_id):
    # print("2-open annotations")
    # print('image_id:%s'%image_id)
    # image_id_1 = image_id.split('/')[-1]
    # print('image_id:%s'%image_id)
    # imade_id = image_id_1.replace("jpg","xml")
    # print('image_id:%s'%image_id)
    # in_file = open('/home/test/darknet/VOC2020/annotations_val_xml/%s.xml'%(image_id))
    # print('infile:','/home/test/darknet/VOC2020/annotations_val_xml/%s'%(image_id))
    in_file = open('/home/jiao/ProjectDirectory/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone-master/datasets1/annotations_val_xml/%s.xml' % (image_id))  ##########
    # print("3-convert to txt")
    out_file = open('/home/jiao/ProjectDirectory/YOLOv3v4-ModelCompression-MultidatasetTraining-Multibackbone-master/datasets1/labels/%s.txt' % (image_id), 'w')  #######
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        # print("write ok")


# wd = getcwd()
wd = " "

for image_set in sets:

    image_ids = open('val.txt').read().strip().split()  ######
    # image_ids = open('%s.txt'%(image_set)).read().strip().split()
    print("start ")
    # list_file = open('%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        # print("again write")
        # print('image_id:%s'%image_id)
        # list_file.write('%s/%s.jpg\n'%(wd, image_id))
        id = image_id.split('/')[-1].replace('jpg', 'xml')
        id = id.split('.')[0]
        print('id:%s' % id)
        convert_annotation(id)
    # list_file.close()