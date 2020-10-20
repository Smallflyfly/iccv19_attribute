#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/10/19
"""
import cv2
import torch
from PIL import Image
from torchvision.transforms import transforms

from model.inception_iccv import InceptionNet

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        normalize
    ])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(pretrained_path):
    model = InceptionNet()
    pretrained_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    new_dict = {}
    for k, _ in model_dict.items():
        raw_name = k.replace('main_branch.', '')
        if raw_name in pretrained_dict:
            new_dict[k] = pretrained_dict[raw_name]
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    return model


def main():
    image = './demo/test.jpg'
    im = Image.open(image).convert('RGB')
    im = transform(im)
    im = im.to(device)
    model = load_trained_model('/weights/')
    model = model.to(device)
    # print(model)
    out = model(im)
    print(out)


if __name__ == '__main__':
    main()