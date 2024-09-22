# ！/usr/bin/env python
# -*-coding:Utf-8 -*-
'''
@Project ：classification_cat_dog
@File    : predict.py
@IDE     ：PyCharm
@Author  ：Huajie Sun
@Time    : 2024/9/18 下午10:09
@Anno    : This is a file about 
'''

import cv2
import torch
from PIL import Image
from torchvision.transforms import transforms

from model import MyResNet18

# device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"


def get_data(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image = cv2.imread(image_path)

    image = transform(image)
    image = image.unsqueeze(0)

    return image


if __name__ == '__main__':
    # 1，model define
    model = MyResNet18(num_classes=2).to(device)  # me

    checkpoints = "checkpoint_cat_dog.pth"  # acc: 0.9048
    model.load_state_dict(torch.load(checkpoints))
    model.eval()

    # data
    image_path = "./test_img/47.jpg"
    image = get_data(image_path)

    # infer
    name_dict = {0: "cat", 1: "dog"}
    outputs = model(image.to(device))
    _, predicted = torch.max(outputs.data, dim=1)
    print(name_dict[predicted.data.item()])