# -*- coding: utf-8 -*-

from ssd.modeling.anchors.prior_box import PriorBox
import torch
import argparse
import os

from models import *


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cfg", type=str,default='V1021.cfg', help="onnx model file")
    parser.add_argument("--model", type=str,default='yolov3_ckpt_69_01051646.pth', help="image file")
    parser.add_argument("--input_size", type=int,default=416, help="input size of image for net")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = GetArgs()
    input_file = args.model
    output_file = "V1023.onnx"
    # cfg.merge_from_file(args.cfg)
    input_size = args.input_size

    # cfg.merge_from_file('configs/mobilenet_v2_ssd320_voc0712.yaml')
    net = Darknet(args.cfg)
    params = torch.load(input_file, map_location=lambda storage, loc: storage)
    net.load_state_dict(params,False)

    device = torch.device("cuda:0") #  if args.use_gpu else "cpu")
    net.eval().to(device)

    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    input_names = ['input']
    # output_names = ['cls_logits', 'bbox_preds', 'anchors']
    output_names = ['cls_and_bbox', 'anchors']
    torch.onnx.export(
        net,
        dummy_input,
        output_file,
        verbose=True,
        input_names=input_names,
        output_names=output_names)
