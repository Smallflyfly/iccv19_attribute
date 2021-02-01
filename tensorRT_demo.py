#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/11/30
"""
import torch
from torchvision.transforms import transforms

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
classes_name = [
    '女性', '16岁以下', '17-30岁', '31-45岁', '体型胖', '体型正常', '体型瘦', '顾客', '职员', '秃头', '长发', '黑发', '带帽', '戴眼镜',
    '围巾', '衬衫', '毛衣', '背心', 'T恤衫', '棉衣', '夹克衫', '西装', '紧身衣', '短袖', '长裤', '裙子', '短裙', '连衣裙', '牛仔裤',
    '紧身裤', '皮鞋', '运动鞋', '靴子', '布鞋', '休闲鞋', '背包', 'SSBag', '手提包', '箱', '塑料袋', '纸袋', '行李箱', '其他东西',
    '打电话', '聊天', '聚会', '保持', '推', '拉', '胳膊扛', '手搬运'
]

if __name__ == '__main__':
    model = InceptionNet()
