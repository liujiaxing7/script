# -*- coding: utf-8 -*-

from ssd.modeling.anchors.prior_box import PriorBox
import torch
import argparse
import os

from models import *


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cfg", type=str,default='cfg/yolov3tiny/yolov3-tiny.cfg', help="onnx model file")
    parser.add_argument("--model", type=str,default='best.pt', help="image file")
    parser.add_argument("--input_size", type=int,default=416, help="input size of image for net")

    args = parser.parse_args()
    return args


class ONNXExportableModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.backbone = model.backbone
        self.predictor = model.box_head.predictor
        self.cfg = model.cfg
        self.cls_headers = self.predictor.cls_headers
        self.reg_headers = self.predictor.reg_headers

    def predict(self, features):
        cls_logits = []
        bbox_pred = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers,
                                                   self.reg_headers):
            cls_logits.append(
                cls_header(feature).permute(0, 2, 3, 1).contiguous())
            bbox_pred.append(
                reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        cls_logits = torch.cat(
            [c.view(c.shape[0], -1, 1, 1) for c in cls_logits], dim=1).view(
                batch_size, -1, 1, self.cfg.MODEL.NUM_CLASSES)
        bbox_pred = torch.cat(
            [l.view(l.shape[0], -1, 1, 1) for l in bbox_pred], dim=1).view(
                batch_size, -1, 1, 4)

        return cls_logits, bbox_pred

    def forward(self, x):
        features = self.backbone(x)
        cls_logits, bbox_preds = self.predict(features)
        cls_and_bbox = torch.cat([cls_logits, bbox_preds], dim=-1)
        priors = PriorBox(self.cfg)()
        return cls_and_bbox, priors


if __name__ == '__main__':
    args = GetArgs()
    input_file = args.model
    output_file = os.path.splitext(args.model)[0] + ".onnx"
    # cfg.merge_from_file(args.cfg)
    input_size = args.input_size

    # cfg.merge_from_file('configs/mobilenet_v2_ssd320_voc0712.yaml')
    net = Darknet(args.cfg)
    params = torch.load(input_file, map_location=lambda storage, loc: storage)['model']
    net.load_state_dict(params,False)

    device = torch.device("cuda:0") #  if args.use_gpu else "cpu")
    net.eval().to(device)

    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    input_names = ['input']
    # output_names = ['cls_logits', 'bbox_preds', 'anchors']
    output_names = ['cls_and_bbox', 'anchors']
    torch.onnx.export(
        ONNXExportableModel(net),
        dummy_input,
        output_file,
        verbose=True,
        input_names=input_names,
        output_names=output_names)
