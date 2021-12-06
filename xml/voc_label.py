import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import argparse
from tqdm import tqdm
import re

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes = {
    "person": 0,
    "person_dummy": 3,
    "escalator": 1,
    "escalator_model": 4,
    "escalator_handrails": 2,
    "escalator_handrails_model": 5
}


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="image file or dir")
    parser.add_argument("--image_set_dir", type=str, help="image file or dir")

    args = parser.parse_args()
    return args


def Walk(path, suffix: list):
    file_list = []
    suffix = [s.lower() for s in suffix]
    if not os.path.exists(path):
        print("not exist path {}".format(path))
        return []

    if os.path.isfile(path):
        return [path, ]

    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower()[1:] in suffix:
                file_list.append(os.path.join(root, file))

    file_list.sort(key=lambda x: int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))

    return file_list


wd = getcwd()
args = GetArgs()
root = args.input
labels_dir = os.path.join(root, 'labels1')
# image_set_dir = os.path.join(root, 'ImageSets/Main/test.txt')
image_set_dir = args.image_set_dir
output_list_file = os.path.join(root, os.path.split(image_set_dir)[-1])
image_dir = os.path.join(root, 'JPEGImages')
anno_dir = os.path.join(root, 'Annotations')


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
    in_file1 = os.path.join(anno_dir, image_id + ".xml")
    print(in_file1)
    if not os.path.exists(in_file1):
        return False
    in_file = open(in_file1)
    txt_file = os.path.join(labels_dir, image_id + ".txt")
    if (not os.path.exists(os.path.dirname(txt_file))):
        os.makedirs(os.path.dirname(txt_file))
    out_file = open(txt_file, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult')
        difficult = difficult.text if difficult is not None else 0

        cls = obj.find('name').text
        if cls not in classes.keys() or int(difficult) == 1:
            continue
        cls_id = classes[cls]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    return True


if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)
if not os.path.exists(image_set_dir):
    image_ids = [os.path.splitext(f)[0][len(anno_dir) + 1:] for f in Walk(anno_dir, ['xml', ])]
else:
    image_ids = open(image_set_dir).read().strip().split()
    # print(image_ids)

list_file = open(output_list_file, 'w')
for image_id in tqdm(image_ids):
    if os.path.exists(image_set_dir):
        image_file = os.path.join(image_dir, image_id + ".jpg")
    else:
        image_file = os.path.join(root, 'images', image_id + ".jpg")
        if not os.path.exists(image_file):
            image_file = os.path.join(root, 'images', image_id + ".png")
    is_save=convert_annotation(image_id)
    # print(is_save)
    if is_save:
        list_file.write(image_file + "\n")

list_file.close()

