import os
from .xml import XML

class VOCDataset(XML):
    class_names = ('__background__',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, data_dir, image_sets_file, target, transform=None, target_transform=None, keep_difficult=False, train=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        year, mode = image_sets_file.split('_')
        data_dir = os.path.join(data_dir, year)
        image_sets_file = os.path.join(data_dir, "ImageSets", "Main", "%s.txt" % mode)

        super(VOCDataset, self).__init__(data_dir, image_sets_file, target, transform, target_transform, keep_difficult, train)


    def get_file(self, index):
        image_id = self.file_list[index]
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        return image_file, annotation_file
