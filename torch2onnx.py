#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/01/22
"""

import torch

from model import inception_iccv
from utils.utils import load_trained_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_model_path = './weights/rap/inception_iccv/40.pth'
onnx_model_path = 'pedestrian_attribute.onnx'


def torch2onnx():
    dummy_input = torch.rand(1, 3, 256, 128, requires_grad=True).cuda()
    model = load_trained_model(torch_model_path)

    model = model.cuda()
    try:
        torch.onnx.export(model, dummy_input, onnx_model_path, verbose=False, training=False, do_constant_folding=False,
                      input_names=['input'], output_names=['output1', 'output2', 'output3', 'output4'], opset_version=11)
    except Exception:
        print("torch转换onnx失败")
        raise Exception


if __name__ == '__main__':
    torch2onnx()