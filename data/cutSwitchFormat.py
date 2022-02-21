# -*- coding: utf-8 -*-
'''
# Created on 02-17-22 16:44
# @Filename: switchFormat.py
# @Desp: 
# @author: xuchang
'''

import os
import jieba 
root_dir = os.path.dirname(__file__)
from c_common import loadFile
labelHead = "__label__"



with open(root_dir+"/label_res.txt","r",encoding="utf8") as f:
    lines = f.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].replace("\r","").replace("\n","")

jieba.load_userdict(root_dir+"/custom_dict.txt") #加载用户词表
stopwords = loadFile(root_dir+"/stopwords.txt")

for i in range(len(lines)):
    sentence = lines[i][:-2]
    label = lines[i][-1]
    segs=jieba.lcut(sentence)    #利用结巴分词进行中文分词
    segs=filter(lambda x:len(x)>1,segs)    #去掉长度小于1的词
    segs=filter(lambda x:x not in stopwords,segs)    #去掉停用词
    lines[i] = " ".join(segs) + " " +label



with open(root_dir+"/ft.train.txt","a",encoding="utf8") as f1:
    for line in lines:
        label = line[-1]
        sentence = line[:-2]
        f1.write(labelHead+str(label)+"\t"+sentence+"\n")


