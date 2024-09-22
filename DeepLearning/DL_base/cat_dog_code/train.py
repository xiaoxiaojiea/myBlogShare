# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
'''
@Project ：classification_cat_dog
@File    : train.py
@IDE     ：PyCharm
@Author  ：Huajie Sun
@Time    : 2024/9/4 下午8:00
@Anno    : This is a file about 
'''
import glob
import os.path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import tqdm

from model import MyResNet18
from dataset import MyDataset

from model_all import resnet101

# device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"


def get_data(root, batch_size, train_rate=0.7, data_split="./data_split.txt"):

    # 首先需要划分数据集
    if not os.path.exists(data_split):
        # 读所有图片的绝对路径
        all_images = list(glob.glob(os.path.join(root, "train/train", "*.jpg")))
        # 得到每个文件对应的标签
        all_labels = []
        for image in all_images:
            if "cat" in os.path.split(image)[-1]:
                all_labels.append(0)
            elif "dog" in os.path.split(image)[-1]:
                all_labels.append(1)

        # 划分出训练与测试并存储
        with open(data_split, 'w') as f:
            for i in range(len(all_images)):
                if i<len(all_images)*train_rate:  # train
                    f.write("train " + all_images[i] +" "+ str(all_labels[i]) + "\n")
                else:  # test
                    f.write("test " + all_images[i] +" "+ str(all_labels[i]) + "\n")


    # 一般这一步是需要自己根据实际数据集定义
    train_dataset = MyDataset(root, "train")
    test_dataset = MyDataset(root, "test")

    # 固定
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()

    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0  # 处理了多少样本
    running_correct = 0  # 正确预测的样本

    train_loader = tqdm.tqdm(train_loader)
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs.to(device))  # 推理

        # 损失计算
        # print(outputs.shape, target.shape)  # torch.Size([32, 10]) torch.Size([32])
        loss = criterion(outputs, target.to(device))

        loss.backward()  # 损失后向传播（当前损失对所有节点求导）
        optimizer.step()  # 梯度更新（使用loss对每个节点计算的梯度进行每个结点的参数更新）

        running_loss += loss.item()  # 累加当前epoch的loss

        _, predicted = torch.max(outputs.data, dim=1)  # 预测最大概率

        # 统计
        running_total += inputs.shape[0]  # 总计数量
        running_correct += (predicted == target.to(device)).sum().item()  # 正确预测数量

    acc = running_correct / running_total
    print("train acc: ", acc)

    torch.save(model.state_dict(), "checkpoint_cat_dog.pth")

    return acc

def test_one_epoch(model, test_loader, epoch):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    acc = correct / total
    print("test acc: ", acc)

    return acc

if __name__ == '__main__':

    # 1，model define
    model = MyResNet18(num_classes=2).to(device)  # me
    # model = resnet101(num_classes=2).to(device)

    # 2，data
    root = "/home/jie/shj_workspace/datasets/for_blog/archive"
    batch_size = 32
    train_loader, test_loader = get_data(root, batch_size)

    # loss
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
    """  适用于分类问题：衡量了预测概率分布与真实概率分布之间的差异
        1，预测与label数据维度（都是概率）： out: torch.Size([B, 10]) label: torch.Size([32])
        2，CEL使用对数损失进行计算，假设有一个包含C(10)个类别的分类问题，并且对于某个样本，真实类别标签概率用yi表示
          （其中i从1到C变化）。每个类别的预测概率由pi表示（同样i从1到C变化），公式如下：
          L = -∑(yi * log(pi))
          求和是针对所有类别进行的，损失值L对错误的预测给予惩罚，当对于真实类别的预测概率较低时，损失值较高。
    """

    # optimizer
    learning_rate = 0.01
    momentum = 0.5
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                momentum=momentum)  # lr学习率，momentum冲量

    # train
    show_acc = True
    epochs = 20
    train_acc_list = []
    test_acc_list = []
    for epoch in range(epochs):
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        train_acc_list.append(train_acc)

        test_acc = test_one_epoch(model, test_loader, epoch)
        test_acc_list.append(test_acc)

