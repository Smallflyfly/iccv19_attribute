#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/10/23
"""

classes = ['Female', 'AgeLess16', 'Age17-30', 'Age31-45', 'BodyFat', 'BodyNormal', 'BodyThin', 'Customer', 'Clerk',
           'BaldHead', 'LongHair', 'BlackHair', 'Hat', 'Glasses', 'Muffler', 'Shirt', 'Sweater', 'Vest', 'TShirt',
           'Cotton', 'Jacket', 'Suit-Up', 'Tight', 'ShortSleeve', 'LongTrousers', 'Skirt', 'ShortSkirt', 'Dress',
           'Jeans', 'TightTrousers', 'LeatherShoes', 'SportShoes', 'Boots', 'ClothShoes', 'CasualShoes', 'Backpack',
           'SSBag', 'HandBag', 'Box', 'PlasticBag', 'PaperBag', 'HandTrunk', 'OtherAttchment', 'Calling', 'Talking',
           'Gathering', 'Holding', 'Pusing', 'Pulling', 'CarryingbyArm', 'CarryingbyHand']

sex = ['Female']
age = ['AgeLess16', 'Age17-30', 'Age31-45']
# 身体胖瘦
body = ['BodyFat', 'BodyNormal', 'BodyThin']




train_list = './dataset/rap/Rap_train_list.txt'

new_train_list = './dataset/rap/train.txt'
with open(train_list) as f:
    pass
