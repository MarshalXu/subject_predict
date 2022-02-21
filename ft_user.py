# -*- coding: utf-8 -*-
'''
# Created on 02-18-22 14:51
# @Filename: ft_user.py
# @Desp: 
# @author: xuchang
'''
from ft_report import ft_predict
import os
import fasttext.FastText as fasttext
root_dir = os.path.dirname(__file__)


model_name = "model/data_dim200_lr00.5_iter20.model" #设置模型文件名称
model_path = os.path.join(root_dir,model_name)
#加载模型
classifier = fasttext.load_model(model_path)

text = ["截至2017年底仍未按规定退还企业",
"关于4个国外贷援款项目2008年度公证审计结果",
"是世行贷款北京环境二期项目的一部分"
]
#预测
for i in range(len(text)):
    res = ft_predict(text[i],classifier=classifier)

    print(res)