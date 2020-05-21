import torch
import torch.nn.functional as F

from backbone import *


class Pyramid_pooling_module(nn.Module):
    def __init__(self, in_channels, depth):
        super(Pyramid_pooling_module, self).__init__()

        self.pool_sizes = [1, 2, 3, 6]

        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(self.pool_sizes[0]),
                                   nn.Conv2d(in_channels, depth, kernel_size=(1, 1)),
                                   nn.BatchNorm2d(depth),
                                   nn.ReLU())
        self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(self.pool_sizes[1]),
                                   nn.Conv2d(in_channels, depth, kernel_size=(1, 1)),
                                   nn.BatchNorm2d(depth),
                                   nn.ReLU())
        self.pool3 = nn.Sequential(nn.AdaptiveAvgPool2d(self.pool_sizes[2]),
                                   nn.Conv2d(in_channels, depth, kernel_size=(1, 1)),
                                   nn.BatchNorm2d(depth),
                                   nn.ReLU())
        self.pool4 = nn.Sequential(nn.AdaptiveAvgPool2d(self.pool_sizes[3]),
                                   nn.Conv2d(in_channels, depth, kernel_size=(1, 1)),
                                   nn.BatchNorm2d(depth),
                                   nn.ReLU())

    def forward(self, x):
        feature1 = self.pool1(x)
        feature1 = F.interpolate(feature1, size=x.size()[-2:], mode='bilinear', align_corners=False)

        feature2 = self.pool2(x)
        feature2 = F.interpolate(feature2, size=x.size()[-2:], mode='bilinear', align_corners=False)

        feature3 = self.pool3(x)
        feature3 = F.interpolate(feature3, size=x.size()[-2:], mode='bilinear', align_corners=False)

        feature4 = self.pool4(x)
        feature4 = F.interpolate(feature4, size=x.size()[-2:], mode='bilinear', align_corners=False)

        pyramid_feature = torch.cat([x, feature1, feature2, feature3, feature4], dim=1)

        return pyramid_feature


class PspNet(nn.Module):
    def __init__(self, backbone, class_num, stride):
        super(PspNet, self).__init__()

        self.backbone = eval(backbone)(pretrained=True, output_stride=stride)
        self.ppm_in_channels = 2048

        depth = self.ppm_in_channels // 4
        self.pyramid_pooling_module = Pyramid_pooling_module(self.ppm_in_channels, depth)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.ppm_in_channels + depth * 4, depth, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(depth),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(depth, class_num, kernel_size=(1, 1))
        )

    def forward(self, input):
        _, out = self.backbone(input)

        out = self.pyramid_pooling_module(out)
        out = self.decoder(out)
        out = F.interpolate(out, size=input.size()[-2:], mode='bilinear', align_corners=False)


        return out
