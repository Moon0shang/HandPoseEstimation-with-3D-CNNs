import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, in_feature, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(in_feature)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(in_feature, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, layer_num, in_feature, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(layer_num):
            layer = _DenseLayer(in_feature + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('dense_layer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, in_feature, out_feature):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(in_feature))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(
            in_feature,
            out_feature,
            kernel_size=1,
            stride=1,
            bias=False
        ))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(2, 4, 4, 2), init_feature=16, bn_size=4, drop_rate=0):
        super(DenseNet, self).__init__()

        # first convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0', nn.Conv3d(
                    in_channels=3,
                    out_channels=init_feature,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False))
            ])
        )
        feature_num = init_feature
        for i, layer_num in enumerate(block_config):
            block = _DenseBlock(layer_num, feature_num, bn_size,
                                growth_rate, drop_rate)
            self.features.add_module('dense_block%d' % (i + 1), block)
            feature_num = feature_num + layer_num * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    in_feature=feature_num,
                    out_feature=feature_num//2
                )
                self.features.add_module('transition%d' % (i + 1), trans)
                feature_num = feature_num // 2

        # final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(feature_num))

        # linear layer
        # self.classifier = nn.Linear(feature_num, class_num)

        # full connect net output features
        self.FC1 = nn.Sequential(
            nn.Linear(170*4*4*4, 4096),
            nn.ReLU(inplace=True),
        )
        self.FC2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
        )
        self.FC3 = nn.Linear(1024, 63)

        # official init from
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        # out = F.relu(features, inplace=True)
        # out = F.avg_pool3d(out, kernel_size=7, stride=1).view(
        #     features.size(0), -1)
        # out = self.classifier(out)
        features = features.view(-1, 170*4*4*4)
        out = self.FC1(features)
        out = F.dropout(out)
        out = self.FC2(out)
        out = F.dropout(out)
        out = self.FC3(out)
        # out = F.linear(170 * 4 * 4 * 4, 4096)
        # out = F.relu(out)
        # out = F.dropout(out)
        # out = F.linear(4096, 1024)
        # out = F.relu(out)
        # out = F.dropout(out)
        # out = F.linear(1024, 21*3)

        return out


if __name__ == "__main__":
    net = DenseNet()
    print(net)
