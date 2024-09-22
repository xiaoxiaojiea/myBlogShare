# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
'''
@Project ：classification_cat_dog
@File    : model.py
@IDE     ：PyCharm
@Author  ：Huajie Sun
@Time    : 2024/9/4 下午7:50
@Anno    : This is a file about 
'''
import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # Conv(without bias)->BN->ReLU
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        # 记录输入，为残差做准备
        identity = x

        # 是否需要进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 卷积1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 卷积2
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差
        out += identity
        out = self.relu(out)

        return out


class MyResNet18(nn.Module):
    def __init__(self, num_classes, block=BasicBlock, blocks_num=[2,2,2,2]):
        super(MyResNet18, self).__init__()

        self.in_channel = 64
        self.groups = 1
        self.width_per_group = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])

        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)

        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)

        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        # print("1: ", x.shape)  # torch.Size([16, 3, 224, 224])

        # -------------- op-1 --------------
        x = self.conv1(x)
        # print("2: ", x.shape)  # torch.Size([16, 64, 112, 112])
        x = self.bn1(x)
        x = self.relu(x)

        # -------------- op-2 --------------
        x = self.maxpool(x)
        # print("3: ", x.shape)  # torch.Size([16, 64, 56, 56])
        x = self.layer1(x)
        # print("4: ", x.shape)  # torch.Size([16, 64, 56, 56])

        # -------------- op-3 --------------
        x = self.layer2(x)
        # print("5: ", x.shape)  # torch.Size([16, 128, 28, 28])

        # -------------- op-4 --------------
        x = self.layer3(x)
        # print("6: ", x.shape)  # torch.Size([16, 256, 14, 14])

        # -------------- op-5 --------------
        x = self.layer4(x)
        # print("7: ", x.shape)  # torch.Size([16, 512, 7, 7])

        # -------------- op-6 - -------------
        x = self.avgpool(x)
        # print("8: ", x.shape)  # torch.Size([16, 512, 1, 1])
        x = torch.flatten(x, 1)
        # print("9: ", x.shape)  # torch.Size([16, 512])
        x = self.fc(x)
        # print("10: ", x.shape)  # torch.Size([16, 2])

        return x

    def _make_layer(self, block, channel, block_num, stride=1):
        # block：BasicBlock； channel：out_channel；block_num：当前层重复的次数；stride：步长

        # 首先判断是否需要进行下采样，如果需要下采样则在这里构建出来下采样层
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:  # 如果要进行下采样
            # 构造下采样层 （虚线的identity）
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        # 当前层的所有layer组成的list（当前层重复的次数）
        layers = []

        # 构建第一个Block（只有第一个Block会进行下采样）
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        # 根据Block个数构建其他Block：除了第一个之外的其他都不用下采样，所以直接相同配置重复即可
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))

        return nn.Sequential(*layers)


if __name__ == '__main__':
    resnet18 = MyResNet18(num_classes=2)
    print(resnet18)

    data = torch.randn([16, 3, 224, 224])
    print(data.shape)

    pred = resnet18(data)
    print(pred.shape)
