import torch.utils.data
import numpy as np

from to_onnx.ssd.structures.container import Container

from to_onnx.ssd.utils.inputs import HDF5Converter
from to_onnx.ssd.utils.inputs import Preprocessor
import cv2


class HDF5Dataset(torch.utils.data.Dataset):
    #  class_names = ('__background__', 'person', 'pet-cat', 'pet-dog',
                   #  'sofa', 'table', 'bed', 'excrement', 'wire', 'key')

    def __init__(self, data_dir, ann_file, transform=None, target_transform=None, remove_empty=False):
        self.h5_dir = data_dir

        self.images, self.images_info, self.labels_info = HDF5Converter.load(
            self.h5_dir)
        self.classes = Preprocessor.classes
        self.id2classes = Preprocessor.generate_id2classes_map(self.classes)

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # load
        image = self.images[index]
        image_info = self.images_info[index]
        label_info = self.labels_info[index]

        # reshape
        im_shape = (image_info[0], image_info[1], 3)
        image = image.reshape(im_shape)
        label_info = label_info.reshape(-1, 5)

        # convert to 3 channels gray image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        boxes = label_info[:, :4].astype(np.float32)
        labels = label_info[:, -1].astype(np.int64)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def get_ground_truth(self):
        return self.labels_info

    def get_annotation(self, index):
        image_id = self.images[index]
        return image_id

    def __len__(self):
        return len(self.images)

    def get_img_info(self, index):
        image_id = self.images[index]
        img_data = self.images_info[image_id]
        return img_data

