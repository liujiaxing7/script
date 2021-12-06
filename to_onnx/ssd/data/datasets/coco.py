import os
import torch.utils.data
import numpy as np
from PIL import Image
import h5py

from to_onnx.ssd.structures.container import Container
import multiprocessing as mp




class COCODataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                   'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                   'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                   'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                   'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self, data_dir, ann_file, transform=None, target_transform=None, remove_empty=False):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.used_classes = ['person']
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.remove_empty = remove_empty
        catIds = self.coco.getCatIds(self.used_classes)
        if self.remove_empty:
            # when training, images without annotations are removed.
            # self.ids = list(self.coco.imgToAnns.keys())
            imgIds = []
            for catId in catIds:
                imgIds.extend(self.coco.getImgIds(catIds=[catId]))
            self.ids = imgIds
        else:
            # when testing, all images used.
            self.ids = list(self.coco.imgs.keys())
        # self.ids = [self.ids[0]]
        coco_categories = sorted(self.coco.getCatIds())
        self.coco_id_to_contiguous_id = {coco_id: i + 1 for i, coco_id in enumerate(catIds)}
        self.coco_id_to_contiguous_id_all = {coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}
        self.use_hdf5 = False
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
            image = self.read_image(image_id)

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

    def _get_annotation(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        ann = self.coco.loadAnns(ann_ids)
        # filter crowd annotations
        ann = [obj for obj in ann if obj["iscrowd"] == 0 and obj['category_id'] in self.coco_id_to_contiguous_id]
        boxes = np.array([self._xywh2xyxy(obj["bbox"]) for obj in ann], np.float32).reshape((-1, 4))
        labels = np.array([self.coco_id_to_contiguous_id[obj["category_id"]] for obj in ann], np.int64).reshape((-1,))
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

    def get_file(self, i):
        return self.ids[i], self.ids[i]

    def read_image(self, image_id):
        file_name = self.coco.loadImgs(image_id)[0]['file_name']
        image_file = os.path.join(self.data_dir, file_name)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def _load_hdf5(self):
        import glob
        hdf5_template = '/home/indemind/Documents/SSD/train_hdf5/train_*.hdf5'
        files = glob.glob(hdf5_template)
        for file in files:
            self.hdf5_buffer[file] = h5py.File(file, 'r', swmr=True, libver='latest')

    def _read_image_from_hdf5(self, image_id):
        num_images_per_file = 5000
        # map image_id to h5 file
        h5_id = image_id//num_images_per_file
        hdf5_template = '/home/indemind/Documents/SSD/train_hdf5/train_{}.hdf5'
        hdf5_path = hdf5_template.format(h5_id)
        image_info = self.coco.loadImgs(image_id)[0]
        height = image_info['height']
        width = image_info['width']
        self.lock.acquire()
        # hwc
        image = self.hdf5_buffer[hdf5_path]['train_images'][image_id%num_images_per_file]
        self.lock.release()
        return image.reshape((height, width, 3))
