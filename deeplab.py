import torch
import torch.nn.functional as F

from backbone import *


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels, depth, output_stride):
        super(AtrousSpatialPyramidPooling, self).__init__()

        if output_stride == 8:
            self.atrous_rates = [1, 12, 24, 36]
        elif output_stride == 16:
            self.atrous_rates = [1, 6, 12, 18]
        else:
            raise AttributeError('output stride is not valid')

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, depth, kernel_size=(1, 1), dilation=self.atrous_rates[0]),
            nn.BatchNorm2d(depth),
            nn.ReLU())
        self.conv_3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, depth, kernel_size=(3, 3), dilation=self.atrous_rates[1], padding=self.atrous_rates[1]),
            nn.BatchNorm2d(depth),
            nn.ReLU())
        self.conv_3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, depth, kernel_size=(3, 3), dilation=self.atrous_rates[2], padding=self.atrous_rates[2]),
            nn.BatchNorm2d(depth),
            nn.ReLU())
        self.conv_3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, depth, kernel_size=(3, 3), dilation=self.atrous_rates[3], padding=self.atrous_rates[3]),
            nn.BatchNorm2d(depth),
            nn.ReLU())
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, depth, (1, 1), bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(depth * 5, depth, kernel_size=(1, 1)),
            nn.BatchNorm2d(depth),
            nn.ReLU())

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        feature1 = self.conv_1x1(x)
        feature2 = self.conv_3x3_1(x)
        feature3 = self.conv_3x3_2(x)
        feature4 = self.conv_3x3_3(x)
        feature5 = self.image_pooling(x)
        feature5 = F.interpolate(feature5, feature1.size()[-2:], mode='bilinear', align_corners=False)

        out = torch.cat([feature1, feature2, feature3, feature4, feature5], dim=1)

        out = self.conv_out(out)
        out = self.dropout(out)
        return out


class DeepLab(nn.Module):
    def __init__(self, backbone, class_num, stride):
        super(DeepLab, self).__init__()

        # backbone
        self.backbone = eval(backbone)(pretrained=True, output_stride=stride)

        self.low_level_feature_in_channel = 256
        self.aspp_in_channel = 2048
        self.depth = 256
        self.low_level_feature_reduction_channel = 48

        self.atrous_spatial_pyramid_pooling = AtrousSpatialPyramidPooling(self.aspp_in_channel,
                                                                          self.depth, stride)

        self.conv_low_level_feature = nn.Sequential(
            nn.Conv2d(self.low_level_feature_in_channel, self.low_level_feature_reduction_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.low_level_feature_reduction_channel),
            nn.ReLU())
        
        self.decoder = nn.Sequential(
            nn.Conv2d(self.low_level_feature_reduction_channel + self.depth, self.depth, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(self.depth),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(self.depth, self.depth, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(self.depth),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(self.depth, class_num, kernel_size=(1, 1))
        )

    def forward(self, input):
        low_level_feature, out = self.backbone(input)
        low_level_feature = self.conv_low_level_feature(low_level_feature)

        out = self.atrous_spatial_pyramid_pooling(out)
        out = F.interpolate(out, size=low_level_feature.size()[-2:], mode='bilinear', align_corners=False)
        out = torch.cat((out, low_level_feature), dim=1)
        out = self.decoder(out)
        out = F.interpolate(out, size=input.size()[-2:], mode='bilinear', align_corners=False)

        return out
