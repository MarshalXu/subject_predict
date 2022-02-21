# -*- coding: utf-8 -*-
'''
# Created on 02-18-22 09:43
# @Filename: splitDataSets.py
# @Desp: 
# @author: xuchang
'''

from math import ceil
import os
from c_common import writeListInFile
import random

root_dir = os.path.dirname(__file__)

with open(os.path.join(root_dir,"ft_total.txt"),"r",encoding="utf8") as f:
    lines = f.readlines()
random.shuffle(lines)

train = lines[:ceil(0.9*len(lines))]

test = lines[ceil(0.9*len(lines)):]

writeListInFile(train,root_dir+"/ft_train.txt")
writeListInFile(test,root_dir+"/ft_test.txt")