import os
import torch.utils.data
import numpy as np
from PIL import Image
import h5py

from to_onnx.ssd.structures.container import Container
import multiprocessing as mp
import cv2
def visualize_bbox(image, anns, size=(600, 800), keep_ratio=True):
    h, w = image.shape[:2]
    for ann in anns:
        box = ann
        cv2.rectangle(
            image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            color=(255, 255, 255),
            thickness=2)
    title = 'detection'

    cv2.imshow(title, image)
    cv2.waitKey(0)





class CleanerMachineDataset(torch.utils.data.Dataset):
    class_names = ('__background__','person', 'pet-cat', 'pet-dog', 'sofa', 'table', 'bed', 'excrement', 'wire', 'key')

    def __init__(self, data_dir, ann_file, transform=None, target_transform=None, remove_empty=False):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.remove_empty = remove_empty

        self.class2_id = {self.class_names[i]:i for i in range(len(self.class_names))}
        if self.remove_empty:
            # when training, images without annotations are removed.
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            # when testing, all images used.
            self.ids = list(self.coco.imgs.keys())
        # self.ids = [self.ids[0]]
        # coco_categories = sorted(self.coco.getCatIds())
        # self.coco_id_to_contiguous_id = {coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}
        # self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}
        self.use_hdf5 = True
        if self.use_hdf5:
            self.hdf5_buffer = {}
            self._load_hdf5()

        self.lock = mp.Lock()


    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        if self.use_hdf5:
            image = self._read_image_from_hdf5(image_id)
        else:
            image = self._read_image(image_id)

        #  import ipdb
        #  ipdb.set_trace()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        #  file_name = self.coco.loadImgs(image_id)[0]['file_name']
        #  visualize_bbox(image, boxes)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        targets = Container(
            boxes=boxes,
            labels=labels,
            file_name=file_name
        )

        return image, targets, index

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    def get_label(self, catId):
        return self.class2_id[self.coco.loadCats(catId)[0]['name']]

    def _get_annotation(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)
        # filter crowd annotations
        ann = [obj for obj in ann if obj["iscrowd"] == 0]
        boxes = np.array([self._xywh2xyxy(obj["bbox"]) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([self.get_label(obj["category_id"]) for obj in ann], np.int64).reshape((-1,))
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        return boxes, labels

    def _xywh2xyxy(self, box):
        x1, y1, w, h = box
        return [x1, y1, x1 + w, y1 + h]

    def get_img_info(self, index):
        image_id = self.ids[index]
        img_data = self.coco.imgs[image_id]
        return img_data

    def _read_image(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        image_file = os.path.join(self.data_dir, file_name)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def _load_hdf5(self):
        import glob
        hdf5_template = os.path.join(self.data_dir, 'train_*.hdf5')
        files = glob.glob(hdf5_template)
        for file in files:
            self.hdf5_buffer[file] = h5py.File(file, 'r', swmr=True, libver='latest')

    def _read_image_from_hdf5(self, image_id):
        num_images_per_file = 5000
        # map image_id to h5 file
        h5_id = image_id//num_images_per_file
        hdf5_template = os.path.join(self.data_dir, 'train_{}.hdf5')
        hdf5_path = hdf5_template.format(h5_id)
        # image_info = self.coco.loadImgs(image_id)[0]
        # height = image_info['height']
        # width = image_info['width']
        self.lock.acquire()
        # hwc
        image = self.hdf5_buffer[hdf5_path]['train_images'][image_id%num_images_per_file]
        image_info = self.hdf5_buffer[hdf5_path]['train_images_info'][image_id%num_images_per_file]
        height = image_info[1]
        width = image_info[0]
        self.lock.release()
        return image.reshape((height, width, 3))
