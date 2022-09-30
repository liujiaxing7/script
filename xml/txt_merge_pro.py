import os

import numpy as np


def read_xml_list(file):
    file_list = []
    with open(file, 'r') as file_r:
        lines = file_r.readlines()
        for line in lines:
            value = line.strip()
            file_list.append(value)
    return file_list
labels1=read_xml_list('/mnt/sdb2/20220825_table/labels2.txt')
labels2=read_xml_list('/mnt/sdb2/20220825_table/labels1.txt')

for i in range(len(labels1)):
    ok=True
    for j in range(len(labels2)):
        if os.path.split(labels1[i])[-1]==os.path.split(labels2[j])[-1]:
            print("ok")
            ok =False
            with open(labels1[i], 'r') as f:
                lb1 = [x.split() for x in f.read().strip().splitlines()]  # labels
            with open(labels2[j], 'r') as f:
                lb2 = [x.split() for x in f.read().strip().splitlines()]
            lb1.extend(lb2)# labels
            label_path = labels1[i].replace('labels2', 'labels3')
            base_path, basename = os.path.split(label_path)
            dst_path = base_path.replace('labels2', 'labels3')
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            with open(label_path, "w+", encoding='UTF-8') as out_file:
                for bb in lb1:
                    out_file.write( " ".join([str(a) for a in bb]) + '\n')
    if ok:
        print("false")
        with open(labels1[i], 'r') as f:
            lb1 = [x.split() for x in f.read().strip().splitlines()]  # labels
        label_path = labels1[i].replace('labels2', 'labels3')
        base_path, basename = os.path.split(label_path)
        dst_path = base_path.replace('labels2', 'labels3')
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        with open(label_path, "w+", encoding='UTF-8') as out_file:
            for bb in lb1:
                out_file.write( " ".join([str(a) for a in bb]) + '\n')