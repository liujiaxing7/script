import os


class DatasetCatalog:
    DATA_DIR = 'datasets/COCO2017'
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2017_val': {
            "data_dir": "val2017",
            "ann_file": "annotations/instances_val2017.json"
        },
        'coco_2017_train': {
            "data_dir": "train2017",
            "ann_file": "annotations/instances_train2017.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        }
    }

    @staticmethod
    def get(name):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif name=='coco_person_bg':
            # attrs = DatasetCatalog.DATASETS[name]
            root_path = '/data/tmp_memory/test_h5_coco'
            args = dict(
                data_dir=root_path,
                ann_file=None,
            )
            return dict(factory="HDF5Dataset", args=args)
        elif name == 'cleaner_machine':
            root_path = '/data/test_images/test_h5'
            args = dict(
                data_dir=root_path,
                ann_file=None,
            )
            return dict(factory="HDF5Dataset", args=args)
        elif name == 'cleaner_machine_person':
            root_path = '/data/test_images/person'
            args = dict(
                data_dir=root_path,
                ann_file=None,
            )
            return dict(factory="HDF5Dataset", args=args)
        elif name == 'cleaner_machine_no_crop':
            root_path = '/data/test_images/test_h5_no_crop'
            args = dict(
                data_dir=root_path,
                ann_file=None,
            )
            return dict(factory="HDF5Dataset", args=args)
        elif name == 'enhancement':
            root_path = '/data/test_images/enhancement'
            args = dict(
                data_dir=root_path,
                ann_file=None,
            )
            return dict(factory="HDF5Dataset", args=args)
        elif name == 'coco_cleaner_machine':
            # attrs = DatasetCatalog.DATASETS[name]
            root_path = '/data/test_images/cleaner_machine'
            args = dict(
                data_dir=root_path,
                ann_file=None,
            )
            return dict(factory="HDF5Dataset", args=args)
        elif "coco" in name:
            coco_root = DatasetCatalog.DATA_DIR
            if 'COCO_ROOT' in os.environ:
                coco_root = os.environ['COCO_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
