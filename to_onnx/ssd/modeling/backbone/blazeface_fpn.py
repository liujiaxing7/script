import torch
import torch.nn as nn
from to_onnx.ssd.modeling import registry
import torch.nn.functional as F

class BlazeBlockV1(nn.Module):
    def __init__(self, in_channels,out_channels,mid_channels=None,stride=1):
        super().__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        # if stride>1:
            # self.use_pool = True
        # else:
            # self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=5,stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
        )

        # if self.use_pool:
            # self.shortcut = nn.Sequential(
                # nn.MaxPool2d(kernel_size=stride, stride=stride),
                # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                # nn.BatchNorm2d(out_channels),
            # )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        # out = (branch1+self.shortcut(x)) if self.use_pool else (branch1+x)
        out = branch1 + x
        return self.relu(out)

class BlazeBlockV2(nn.Module):
    def __init__(self, in_channels,out_channels,mid_channels=None,stride=1):
        super().__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        # if stride>1:
            # self.use_pool = True
        # else:
            # self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=5,stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
        )

        # if self.use_pool:
        self.shortcut = nn.Sequential(
            nn.MaxPool2d(kernel_size=stride, stride=stride),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        # out = (branch1+self.shortcut(x)) if self.use_pool else (branch1+x)
        out = branch1 + self.shortcut(x)
        return self.relu(out)

class DoubleBlazeBlockV1(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,stride=1):
        super().__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        # if stride > 1:
            # self.use_pool = True
        # else:
            # self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        # if self.use_pool:
            # self.shortcut = nn.Sequential(
                # nn.MaxPool2d(kernel_size=stride, stride=stride),
                # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                # nn.BatchNorm2d(out_channels),
            # )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        # out = (branch1 + self.shortcut(x)) if self.use_pool else (branch1 + x)
        out = branch1 + x
        return self.relu(out)

class DoubleBlazeBlockV2(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,stride=1):
        super().__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        # if stride > 1:
            # self.use_pool = True
        # else:
            # self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        # if self.use_pool:
        self.shortcut = nn.Sequential(
            nn.MaxPool2d(kernel_size=stride, stride=stride),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        # out = (branch1 + self.shortcut(x)) if self.use_pool else (branch1 + x)
        out = branch1 + self.shortcut(x)
        return self.relu(out)


class BlazeFaceFPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BlazeBlockV1(in_channels=24, out_channels=24),
            BlazeBlockV1(in_channels=24, out_channels=24))

        self.layer2 = nn.Sequential(
            BlazeBlockV2(in_channels=24, out_channels=48, stride=2),
            BlazeBlockV1(in_channels=48, out_channels=48),
            BlazeBlockV1(in_channels=48, out_channels=48),
        )

        self.layer3 = nn.Sequential(
            DoubleBlazeBlockV2(in_channels=48, out_channels=96, mid_channels=24, stride=2),
            DoubleBlazeBlockV1(in_channels=96, out_channels=96, mid_channels=24),
            DoubleBlazeBlockV1(in_channels=96, out_channels=96, mid_channels=24),

        )
        self.layer4 = nn.Sequential(DoubleBlazeBlockV2(in_channels=96, out_channels=96, mid_channels=24, stride=2),
            DoubleBlazeBlockV1(in_channels=96, out_channels=96, mid_channels=24),
            DoubleBlazeBlockV1(in_channels=96, out_channels=96, mid_channels=24))

        self.extras = nn.ModuleList([
            DoubleBlazeBlockV2(in_channels=96, out_channels=96, mid_channels=24, stride=2),
            DoubleBlazeBlockV2(in_channels=96, out_channels=96, mid_channels=24, stride=2),
            DoubleBlazeBlockV2(in_channels=96, out_channels=96, mid_channels=24, stride=2),
        ])
        self.lateral_channels = 24
        self.lateral_layers = nn.ModuleList([nn.Conv2d(96, self.lateral_channels, 1, 1, 0)]*5)
        #  self.initialize()

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        num_layers = 5
        results = []
        for layer_ind in range(num_layers):
            x = getattr(self, 'layer{}'.format(layer_ind))(x)
            if layer_ind>=3:
                results.append(x)
        for i in range(len(self.extras)):
            x = self.extras[i](x)
            results.append(x)

        c1, c2, c3, c4, c5 = results
        p5 = self.lateral_layers[4](c5)
        p4 = self.upsample_add(p5, self.lateral_layers[3](c4))
        p3 = self.upsample_add(p4, self.lateral_layers[2](c3))
        p2 = self.upsample_add(p3, self.lateral_layers[1](c2))
        p1 = self.upsample_add(p2, self.lateral_layers[0](c1))

        rcnn_feature_maps = [p1, p2, p3, p4, p5]
        return rcnn_feature_maps

@registry.BACKBONES.register('blazeface_fpn')
def blazeface_fpn(cfg, pretrained=True):
    model = BlazeFaceFPN()
    return model

if __name__=='__main__':
    model = BlazeFaceFPN()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    # out = model(input)
    torch.onnx.export(model, input, "blazeface.onnx", verbose=True)
    # print(out.shape)
