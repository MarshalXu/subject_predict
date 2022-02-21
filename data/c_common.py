# -*- coding: utf-8 -*-
'''
# Created on 02-17-22 15:11
# @Filename: c_common.py
# @Desp: 
# @author: xuchang
'''

import os

root_dir = os.path.dirname(__file__)

def loadFile(path):
    with open(path,mode = "r",encoding="utf8") as f:
        lines = f.readlines()
    return lines



def writeListInFile(lines,path,mode = "a"):
    with open(path,mode = mode,encoding="utf8") as f:
        for line in lines:
            f.write(line)