import os

import cv2


def get_xml(input_dir):
    xml_path_list = []
    for (root_path, dirname, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.png'):
                xml_path = root_path + "/" + filename
                xml_path_list.append(xml_path)
    return xml_path_list


import argparse
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--area', type=int, default="468")
    parser.add_argument('--input_dir', type=str, default="/mnt/sdb1/Data/data3/902_0712/7_9/images/val")
    parser.add_argument('--output_dir', type=str,
                        default="/mnt/sdb1/Data/data3/902_0712/7_9/images/light1")
    args = parser.parse_args()

    xml_path_list = get_xml(args.input_dir)
    for xml_file in tqdm(xml_path_list):
        im1=cv2.imread(xml_file,cv2.IMREAD_GRAYSCALE)

        im2=cv2.equalizeHist(im1)

        output_file = os.path.join(args.output_dir, xml_file[len(args.input_dir) + 1:])
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))

        cv2.imwrite(output_file,im2)

