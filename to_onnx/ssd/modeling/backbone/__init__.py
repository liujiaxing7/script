from .. import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet
from .blazeface import BlazeFace
from .blazeface_fpn import BlazeFaceFPN
from .mobilenet_fpn import MobileNetFPNV2
from .mobilenet_fpn_old import MobileNetFPNV2Old

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet', 'BlazeFace', 'BlazeFaceFPN', 'MobileNetFPNV2', 'MobileNetFPNV2Old']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
