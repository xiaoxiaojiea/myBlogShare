# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
'''
@Project ：classification_cat_dog
@File    : dataset.py
@IDE     ：PyCharm
@Author  ：Huajie Sun
@Time    : 2024/9/3 下午8:00
@Anno    : This is a file about
'''

import glob
import os.path
import random

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class MyDataset(Dataset):
    def __init__(self, root, split="train", transform=None, data_split="./data_split.txt"):
        self.root = root
        self.split = split
        self.transform = transform

        # 图像以及标签list
        self.all_images = []
        self.all_labels = []  # cat: 0, dog: 1

        # 读文件
        with open(data_split, "r") as f:
            all_lines = f.readlines()

            for i in range(len(all_lines)):
                cur_line = all_lines[i].split(" ")
                if cur_line[0] == self.split:
                    self.all_images.append(cur_line[1])
                    self.all_labels.append(int(cur_line[2]))

        if self.transform == None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __getitem__(self, item):
        # 读取图像
        image_path = self.all_images[item]
        image = cv2.imread(image_path)
        # 图像增强
        if self.transform is not None:
            image = self.transform(image)

        # 读取标签
        label = self.all_labels[item]

        return image, label


    def __len__(self):
        return len(self.all_images)


if __name__ == '__main__':

    # # train
    # root = "/home/jie/shj_workspace/datasets/for_blog/archive"
    # split = "train"
    #
    # train_dataset = MyDataset(root, split)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    #
    # for i, data in enumerate(train_dataloader):
    #     image, label = data
    #
    #     print(image.shape)
    #     print(label.shape)

    # test
    root = "/home/jie/shj_workspace/datasets/for_blog/archive"
    split = "test"

    train_dataset = MyDataset(root, "train")
    print(len(train_dataset))

    test_dataset = MyDataset(root, "test")
    print(len(test_dataset))


    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

    for i, data in enumerate(test_loader):
        image, label = data

        print(image.shape)
        print(label.shape)



