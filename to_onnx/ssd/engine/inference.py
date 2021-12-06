import logging
import os

import torch
import torch.utils.data
from tqdm import tqdm

from to_onnx.ssd.data.build import make_data_loader
from to_onnx.ssd.data.datasets.evaluation import evaluate

from to_onnx.ssd.utils import dist_util, mkdir
from to_onnx.ssd.utils.dist_util import synchronize, is_main_process
import cv2
import numpy as np
import copy
from to_onnx.ssd.data.build import concate_dataset


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = dist_util.all_gather(predictions_per_gpu)
    if not dist_util.is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("SSD.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def compute_on_dataset(model, data_loader, device):
    results_dict = {}
    for batch in tqdm(data_loader):
        images, targets, image_ids = batch
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            outputs = model(images.to(device))

            outputs = [o.to(cpu_device) for o in outputs]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, outputs)}
        )
    return results_dict

def Show(input_size, predictions, dataset, wait=0):
    for i, p in enumerate(predictions):
        p = {k:p[k].cpu().numpy() for k in p}
        img_file, ann_file = dataset.get_file(i)
        gt_boxes, gt_labels = dataset.get_annotation(ann_file)
        image = dataset.read_image(img_file)
        dst = cv2.resize(image, (input_size, input_size))

        for box, label, score in zip(p['boxes'], p['labels'], p['scores']):
            x, y, x2, y2 = box
            cv2.rectangle(dst, (x, y), (x2, y2), (0, 0, 255))
            cv2.putText(dst, dataset.class_names[label]+" {:.2f}".format(score), (max(0, int(x)), max(15, int(y)+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        image = cv2.resize(dst, (image.shape[1], image.shape[0]))
        for box, label in zip(gt_boxes, gt_labels):
            x, y, x2, y2 = box
            cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0))
            # cv2.putText(image, dataset.class_names[label], (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        image = cv2.resize(image, (int(image.shape[1]/image.shape[0]*960), 960))
        cv2.imshow("draw", image)
        cv2.waitKey(wait)

def inference(model, data_loader, dataset_name, device, output_folder=None, use_cached=False, **kwargs):
    dataset = data_loader.dataset
    logger = logging.getLogger("SSD.inference")
    logger.info("Evaluating {} dataset({} images):".format(dataset_name, len(dataset)))
    predictions_path = os.path.join(output_folder, 'predictions.pth')
    if use_cached and os.path.exists(predictions_path):
        predictions = torch.load(predictions_path, map_location='cpu')
    else:
        predictions = compute_on_dataset(model, data_loader, device)
        synchronize()
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    if not is_main_process():
        return
    if output_folder and not (use_cached and os.path.exists(predictions_path)):
        torch.save(predictions, predictions_path)
    return predictions

def get_path(output_dir, model_name, dataset_name):
    folder = os.path.join(output_dir, model_name, os.path.splitext(dataset_name)[0])
    if not os.path.exists(folder):
        mkdir(folder)
    return folder

@torch.no_grad()
def forward_all_data(cfg, model, data_loaders_val, output_dir, model_name, **kwargs):
    logger = logging.getLogger("SSD.inference")

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()

    device = torch.device(cfg.MODEL.DEVICE)
    predictions_list = []
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = get_path(output_dir, model_name, dataset_name)
        predictions = inference(model, data_loader, dataset_name, device, output_folder, True, **kwargs)
        predictions_list.append(predictions)
    return predictions_list

@torch.no_grad()
def do_evaluation(cfg, output_dir, model, distributed, wait=-1, model_name="model", **kwargs):
    output_dir = os.path.join(output_dir, 'test')
    input_size = cfg.INPUT.IMAGE_SIZE

    data_loaders_val = make_data_loader(cfg, is_train=False, distributed=distributed)
    eval_results = []
    predictions_list = forward_all_data(cfg, model, data_loaders_val, output_dir, model_name, **kwargs)
    predictions_mixed = np.concatenate(copy.deepcopy(predictions_list)).tolist()

    logger = logging.getLogger("SSD.inference")
    logger.info("evaluate dataset {}...".format("all"))
    output_folder = get_path(output_dir, model_name, "all")
    eval_result, threshold = evaluate(dataset=concate_dataset(copy.deepcopy(data_loaders_val)), predictions=predictions_mixed,
                                      output_dir=output_folder, save_anno=False, **kwargs)

    for dataset_name, data_loader, predictions in zip(cfg.DATASETS.TEST, data_loaders_val, predictions_list):
        logger.info("evaluate dataset {}...".format(dataset_name))
        output_folder = get_path(output_dir, model_name, dataset_name)
        if wait >= 0:
            Show(input_size, predictions, data_loader.dataset, wait=wait)
        eval_result, _ = evaluate(dataset=data_loader.dataset, predictions=predictions, output_dir=output_folder,
                                  save_anno=cfg.TEST.SAVE_ANNO, threshold=threshold, **kwargs)
        eval_results.append(eval_result)
    return eval_results
