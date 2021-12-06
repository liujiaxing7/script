import os
from to_onnx.ssd.utils.inputs import Preprocessor
from .xml import XML

class ImageByXML(XML):
    class_names = Preprocessor.classes

    def __init__(self, data_dir, image_sets_file, target, transform=None, target_transform=None, keep_difficult=False, train=False):
        super(ImageByXML, self).__init__(data_dir, image_sets_file, target, transform, target_transform, keep_difficult, train)

    def get_file(self, index):
        image_id = self.file_list[index]
        return os.path.join(self.data_dir, image_id), self._ann_file(image_id)
