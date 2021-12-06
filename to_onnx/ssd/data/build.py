import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from to_onnx.ssd.data import samplers
from to_onnx.ssd.data.datasets import build_dataset
from to_onnx.ssd.data.transforms import build_transforms, build_target_transform
from to_onnx.ssd.structures.container import Container

from prefetch_generator import BackgroundGenerator


class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            targets = Container(
                {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
            )
        else:
            targets = None
        return images, targets, img_ids


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__(), max_prefetch=128)


# class DataPrefetcher():
    # def __init__(self, loader):
        # self.loader = loader

    # def prefetch(self):
        # try:
            # self.next_input, self.
        # pass

class data_prefetcher():
    def __init__(self, loader):
        self.size = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        device = torch.device('cuda')
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.to(device, non_blocking=True)
            self.next_target = self.next_target.to(device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target["boxes"].record_stream(torch.cuda.current_stream())
            target["labels"].record_stream(torch.cuda.current_stream())
        if input is not None and target is not None:
            self.preload()
        return input, target, None

def check_dataset(dataset):
    import sys
    num = len(dataset)
    for i in range(num):
        sample = dataset[i]
        mean1 = dataset.dur1/(i+1)
        mean2 = dataset.dur2/(i+1)
        sys.stdout.write('\rinput_shape: {}, read_time: {}, preprocess_time: {}, {}/{}'.format(dataset.shape, mean1, mean2, i, num))

def concate_dataset(dataloaders:list):
    length = len(dataloaders)
    if length == 0:
        return None
    elif length == 1:
        return dataloaders[0].dataset
    all_data = dataloaders[0].dataset
    for i in range(1, length):
        all_data.concate(dataloaders[i].dataset)

    return all_data

def make_data_loader(cfg, is_train=True, distributed=False, max_iter=None, start_iter=0):
    train_transform = build_transforms(cfg, is_train=is_train)
    target_transform = build_target_transform(cfg) if is_train else None
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(dataset_list, cfg.DATASETS.TYPE, cfg.DATASETS.ROOT, cfg.DATASETS.MIXUP.OBJECT, train_transform, target_transform, is_train)

    # check_dataset(datasets[0])

    shuffle = is_train or distributed
    shuffle = True

    data_loaders = []

    for dataset in datasets:
        if distributed:
            sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=True if is_train else False)
        if max_iter is not None:
            batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter, start_iter=start_iter)

        data_loader = DataLoaderX(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train))
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_prefetcher(data_loaders[0])
    return data_loaders
