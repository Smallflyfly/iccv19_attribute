#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/10/20
"""
from scipy.io import loadmat


def main():
    data = loadmat('./dataset/rap/RAP_annotation/RAP_annotation.mat')
    print(data)
    dataset = dict()
    dataset['att_name'] = []
    for idx in range(51):
        dataset['att_name'].append(data['RAP_annotation'][0][0][6][idx][0][0])
    print(dataset['att_name'])
    print(len(dataset['att_name']))


if __name__ == '__main__':
    main()