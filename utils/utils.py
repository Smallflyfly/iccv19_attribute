#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2021/02/01
"""
import torch

from model.inception_iccv import InceptionNet, inception_iccv


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
    model = inception_iccv(pretrained=False)
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
