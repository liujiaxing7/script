from torch import nn
import torch.nn.functional as F

from to_onnx.ssd.modeling import registry
from to_onnx.ssd.utils.model_zoo import load_state_dict_from_url

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetFPNV2Old(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.extras = nn.ModuleList([
            InvertedResidual(1280, 512, 2, 0.2),
            InvertedResidual(512, 256, 2, 0.25),
            InvertedResidual(256, 256, 2, 0.5),
        ])

        self.lateral1 = nn.Conv2d(32,256,  1, 1, 0)
        self.lateral2 = nn.Conv2d(96, 256, 1, 1, 0)
        self.lateral3 = nn.Conv2d(1280, 256, 1, 1, 0)
        self.lateral4 = nn.Conv2d(512, 256, 1, 1, 0)
        self.lateral5 = nn.Conv2d(256, 256, 1, 1, 0)
        self.lateral6 = nn.Conv2d(256, 256, 1, 1, 0)

        self.smooth1 = nn.Conv2d(256, 256, 3,1,1)
        self.smooth2 = nn.Conv2d(256, 256, 3,1,1)
        self.smooth3 = nn.Conv2d(256, 256, 3,1,1)
        self.smooth4 = nn.Conv2d(256, 256, 3,1,1)
        self.smooth5 = nn.Conv2d(256, 256, 3,1,1)
        self.reset_parameters()

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def reset_parameters(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        features = []
        for i in range(14):
            x = self.features[i](x)
            if i==6:
                features.append(x)
        features.append(x)

        for i in range(14, len(self.features)):
            x = self.features[i](x)
        features.append(x)

        for i in range(len(self.extras)):
            x = self.extras[i](x)
            features.append(x)

        C1, C2, C3, C4, C5, C6 = features
        P6 = self.lateral6(C6)
        P5 = self.upsample_add(P6, self.lateral5(C5))
        P4 = self.upsample_add(P5, self.lateral4(C4))
        P3 = self.upsample_add(P4, self.lateral3(C3))
        P2 = self.upsample_add(P3, self.lateral2(C2))
        P1 = self.upsample_add(P2, self.lateral1(C1))


        P1 = self.smooth1(P1)
        P2 = self.smooth2(P2)
        P3 = self.smooth3(P3)
        P4 = self.smooth4(P4)
        P5 = self.smooth5(P5)

        #  return tuple(features)
        rcnn_feature_maps = [P1, P2, P3, P4, P5, P6]
        return rcnn_feature_maps


@registry.BACKBONES.register('mobilenet_fpn_old')
def mobilenet_fpn_old(cfg, pretrained=True):
    model = MobileNetFPNV2Old()
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['mobilenet_v2']), strict=False)
    return model
