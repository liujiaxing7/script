from torch.utils.data import ConcatDataset

from .voc import VOCDataset
from .coco import COCODataset
from .cleaner_machine import CleanerMachineDataset
from .hdf5 import HDF5Dataset
from .image_by_xml import ImageByXML
import os

_DATASETS = {
    'VOCDataset': VOCDataset,
    'COCODataset': COCODataset,
    'CleanerMachineDataset': CleanerMachineDataset,
    'HDF5Dataset': HDF5Dataset,
    'ImageByXML': ImageByXML
}

def get_args(root_dir, dataset_name, target, transform, target_transform, is_train, factory):
    args = {'data_dir': root_dir, }
    args['transform'] = transform
    args['target_transform'] = target_transform
    if factory == VOCDataset:
        args['keep_difficult'] = not is_train
        args['image_sets_file'] = dataset_name
        args['train'] = is_train
        args['target'] = target
    elif factory == COCODataset:
        args['data_dir'] = os.path.join(root_dir, os.path.splitext(dataset_name)[0].split('_')[-1])
        args['remove_empty'] = is_train
        args['ann_file'] = os.path.join(root_dir, dataset_name)
    elif factory == ImageByXML:
        args['image_sets_file'] = os.path.join(root_dir, dataset_name)
        args['keep_difficult'] = not is_train
        args['train'] = is_train
        args['target'] = target

    return  args

def build_dataset(dataset_list, dataset_type, root_dir, target, transform=None, target_transform=None, is_train=True):
    assert len(dataset_list) > 0
    datasets = []
    for dataset_name in dataset_list:
        factory = _DATASETS[dataset_type]
        args = get_args(root_dir, dataset_name, target, transform, target_transform, is_train, factory)

        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0] # todo
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]
