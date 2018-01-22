# Based on Stanford ECG detector

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo

__all__ = ['stanford_net', 'stanford_selu', 'stanford_18', 'stanford_20']

def conv(in_channels, out_channels, stride=1, dilation=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=15,
                     stride=stride,
                     padding=7*dilation,
                     bias=False,
                     dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation_factor=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(inplanes, planes, stride, dilation=dilation_factor)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes, dilation=dilation_factor)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreactBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,
                 dilation_factor=1, act=nn.ReLU(inplace=True)):
        super(PreactBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act = act
        use_selu = type(act) == torch.nn.modules.activation.SELU
        self.drop1 = nn.AlphaDropout() if use_selu else nn.Dropout()
        self.conv1 = conv(in_channels, out_channels, stride, dilation=1)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.AlphaDropout() if use_selu else nn.Dropout()
        self.conv2 = conv(out_channels, out_channels, dilation=dilation_factor)
        self.downsample = nn.MaxPool1d(stride, ceil_mode=True) if stride != 1 else None
        self.skip_connection = lambda x: x
        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, bias=False))
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)
        residual = self.skip_connection(residual)

        x = self.bn1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.act(x)
        x = self.drop2(x)
        x = self.conv2(x)

        x += residual
        x = self.act(x)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, inplanes=64, num_classes=3, in_channels=1,
                 dilated=False, selu=False):
        self.dilation_factor = 2 if dilated else 1
        self.inplanes = inplanes
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size=15,
                               stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.act = nn.SELU(inplace=True) if selu else nn.ReLU(inplace=True)

        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        self.conv2 = nn.Conv1d(self.inplanes, self.inplanes, kernel_size=15, stride=1, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(self.inplanes)
        self.drop1 = nn.AlphaDropout() if selu else nn.Dropout()
        self.conv3 = nn.Conv1d(self.inplanes, self.inplanes, 15, padding=7, bias=False, dilation=self.dilation_factor)


        self.layer1 = self._make_layer(block, inplanes*1, layers[0], stride=2)
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes*3, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes*4, layers[3], stride=2)
        self.num_features = inplanes * 4 * block.expansion
        final_conv = nn.Conv1d(self.num_features, num_classes, 1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(inplanes*4*block.expansion),
            self.act,
            final_conv,
            nn.AdaptiveAvgPool1d(1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes))
        self.inplanes = planes * block.expansion

        for i in range(blocks - 1):
            if i % 2 == 0:
                layers.append(block(
                    planes, planes, dilation_factor=self.dilation_factor,
                    stride=stride, act=self.act))
            else:
                layers.append(block(
                    planes, planes, dilation_factor=self.dilation_factor,
                    act=self.act))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        residual = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x += residual

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)
        return x.squeeze(-1)

def stanford_net(pretrained=False, **kwargs):
    model = ResNet(PreactBlock, [3, 4, 4, 4], **kwargs)
    return model

def stanford_selu(pretrained=False, **kwargs):
    model = ResNet(PreactBlock, [3, 4, 4, 4], selu=True, **kwargs)
    return model

def stanford_18(pretrained=False, **kwargs):
    model = ResNet(PreactBlock, [3, 4, 4, 6], **kwargs)
    return model

def stanford_20(pretrained=False, **kwargs):
    model = ResNet(PreactBlock, [3, 4, 6, 6], **kwargs)
    return model
