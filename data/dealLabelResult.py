# -*- coding: utf-8 -*-
'''
# Created on 02-18-22 16:08
# @Filename: dealLabelResult.py
# @Desp: 
# @author: xuchang
'''
import os
import jieba 
import random
from math import ceil
root_dir = os.path.dirname(__file__)
from c_common import loadFile,writeListInFile
labelHead = "__label__"
train_file_path = os.path.join(root_dir,"ft_train.txt")
test_file_path = os.path.join(root_dir,"ft_test.txt")

with open(root_dir+"/labelresult.txt","r",encoding="utf8") as f:
    lines = f.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].replace("\r","").replace("\n","")

jieba.load_userdict(root_dir+"/custom_dict.txt") #加载用户词表
stopwords = loadFile(root_dir+"/stopwords.txt")

for i in range(len(lines)):

    sentence = lines[i].split("\t",1)[-1]
    label = lines[i].split("\t",1)[0]
    segs=jieba.lcut(sentence)    #利用结巴分词进行中文分词
    segs=filter(lambda x:len(x)>1,segs)    #去掉长度小于1的词
    segs=filter(lambda x:x not in stopwords,segs)    #去掉停用词
    lines[i] = label + "\t" + " ".join(segs) +"\n"

random.shuffle(lines)

cut_pos = ceil(0.9*len(lines))

train_lines = lines[:cut_pos]
test_lines = lines[cut_pos:]

writeListInFile(train_lines,train_file_path)
writeListInFile(test_lines,test_file_path)