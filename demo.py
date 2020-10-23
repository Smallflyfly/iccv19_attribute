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
import numpy as np

from model.inception_iccv import InceptionNet

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(size=(256, 128)),
    transforms.ToTensor(),
    normalize
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['Female', 'AgeLess16', 'Age17-30', 'Age31-45', 'BodyFat', 'BodyNormal', 'BodyThin', 'Customer', 'Clerk',
           'BaldHead', 'LongHair', 'BlackHair', 'Hat', 'Glasses', 'Muffler', 'Shirt', 'Sweater', 'Vest', 'TShirt',
           'Cotton', 'Jacket', 'Suit-Up', 'Tight', 'ShortSleeve', 'LongTrousers', 'Skirt', 'ShortSkirt', 'Dress',
           'Jeans', 'TightTrousers', 'LeatherShoes', 'SportShoes', 'Boots', 'ClothShoes', 'CasualShoes', 'Backpack',
           'SSBag', 'HandBag', 'Box', 'PlasticBag', 'PaperBag', 'HandTrunk', 'OtherAttchment', 'Calling', 'Talking',
           'Gathering', 'Holding', 'Pusing', 'Pulling', 'CarryingbyArm', 'CarryingbyHand']


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_trained_model(pretrained_path):
    model = InceptionNet()
    model_dict = torch.load(pretrained_path)
    # print(model_dict)
    # fang[-1]
    # model_dict = model_dict['state_dict']
    # for key in model_dict:
    #     print(key)
    # print(type(model_dict))
    if "state_dict" in model_dict.keys():
        model_dict = remove_prefix(model_dict['state_dict'], 'module.')

    # model_dict = model.state_dict()
    # new_dict = {}
    # for k, _ in model_dict.items():
    #     raw_name = k.replace('main_branch.', '')
    #     if raw_name in pretrained_dict:
    #         new_dict[k] = pretrained_dict[raw_name]
    # model_dict.update(new_dict)
    check_keys(model, model_dict)
    model.load_state_dict(model_dict)
    return model


def main():
    image = './dataset/demo/demo_image.png'
    im = Image.open(image).convert('RGB')
    im = transform(im)
    im = im.to(device)
    model = load_trained_model('./weights/rap/inception_iccv/40.pth')
    model = model.to(device)
    # print(model)
    im = im.unsqueeze(0)
    output = model(im)
    output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
    output = torch.sigmoid(output).cpu().detach().numpy()
    output = np.where(output > 0.7, 1, 0)
    out_list = output[0].tolist()
    print(out_list)
    pred = torch.from_numpy(output).long()
    # print(output.shape)
    # print(output)
    target = np.array(
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
         0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]).astype(np.long)
    target = np.array(
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.long)
    target = torch.from_numpy(target).long()
    correct = pred.eq(target)
    correct = correct.numpy()

    attr_num = 51
    batch_size = 1

    res = []
    for k in range(attr_num):
        res.append(1.0 * sum(correct[:, k]) / batch_size)

    print(sum(res) / attr_num)

    for i in range(51):
        if out_list[i] == 1:
            print(classes[i])


if __name__ == '__main__':
    main()
