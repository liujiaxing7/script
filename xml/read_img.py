import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
path='/media/fandong/U393/BASE/20211020_single_laybox'
path1='rgb'
# path2='cam0'
output = "output"
import time
from tqdm import tqdm

def get_xml(input_dir):
    xml_path_list = []
    for (root_path, dirname, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                xml_path = root_path + "/" + filename
                xml_path_list.append(xml_path)
    return xml_path_list
def readimg():
    array_img = []
    img_file=get_xml(path)
    img_file.sort()

    for i,f in enumerate(img_file):
        print(f)
        if i%3==0:
            image=cv2.imread(f,cv2.IMREAD_GRAYSCALE)
            equ_image=cv2.equalizeHist(image)
            # image=resizeimg(img,1)-215:Assertion failed) _src.type() == CV_8UC1 in function 'equalizeHist
            array_img.append(equ_image)
            if len(array_img)==6:
                imshowimg(array_img, f)
                array_img = []


#readimg()
def imshowimg(array_img,image_path):
    image1 = np.hstack(array_img[0:3])
    image2 = np.hstack(array_img[3:6])
    image = np.vstack((image1, image2))
    image = cv2.resize(image,(1720,960))
    cv2.namedWindow(image_path)
    cv2.imshow(image_path, image)
    cv2.waitKey(0)
    cv2.destroyWindow(image_path)

    return


def resizeimg(img,scale):
    h,w=img.shape[0],img.shape[1]
    new_h=int(h*scale)
    new_w=int(w*scale)
    image=cv2.resize(img,(new_w,new_h))
    return image

def main():
    readimg()

if __name__ == '__main__':
    main()