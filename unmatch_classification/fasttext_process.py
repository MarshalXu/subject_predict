# -*- coding: utf-8 -*-
'''
# Created on 02-17-22 14:11
# @Filename: fasttext_process.py
# @Desp: 
# @author: xuchang
'''
from math import ceil
import os
import pandas as pd
import random
import fasttext
import jieba
from sklearn.model_selection import train_test_split
from c_common import loadFile

root_dir = os.path.dirname(__file__)

cate_dic = {"政府":0,"被审计单位":1}

dataname = os.path.join(root_dir,"data/label_res.txt")
print(dataname)
def tmporary_preprocess():
    """
    # 暂时的语料处理 后面可能不用这个
    """
    lines = loadFile(dataname)
    for line in lines:
        # print(line)
        line = line.replace("\r","").replace("\n","").replace("\t","")
        line = line[0:-1] + " __label__" + line[-1]
        
        with open(root_dir+"/data/ft_data.txt","a",encoding="utf8") as f:
            f.write(line + "\n")


def preprocess_text(content_line,sentences,category,stopwords):
    for line in content_line:
        try:
            segs=jieba.lcut(line)    #利用结巴分词进行中文分词
            segs=filter(lambda x:len(x)>1,segs)    #去掉长度小于1的词
            segs=filter(lambda x:x not in stopwords,segs)    #去掉停用词
            sentences.append("__lable__"+str(category)+" , "+" ".join(segs))    #把当前的文本和对应的类别拼接起来，组合成fasttext的文本格式
        except Exception as e:
            print (line)
            continue

"""
函数说明：把处理好的写入到文件中，备用
参数说明：

"""
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

# print_results(*model.test('test.txt'))


if __name__ == "__main__":
    sentences = []
    # tmporary_preprocess()
    jieba.load_userdict(root_dir+"/data/custom_dict.txt")
    stopwords = loadFile(root_dir+"/data/stopwords.txt")
    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].replace("\r","").replace("\n","")
    data = loadFile(root_dir+"/data/ft_data.txt")
    for line in data:
        line = line.replace("\r","").replace("\n","")
        segs=jieba.lcut(line[:-11])    #利用结巴分词进行中文分词
        segs=filter(lambda x:len(x)>1,segs)    #去掉长度小于1的词
        segs=filter(lambda x:x not in stopwords,segs)    #去掉停用词
        category = line[-1]
        # sentences.append("__lable__"+str(category)+" , "+" ".join(segs))    #把当前的文本和对应的类别拼接起来，组合成fasttext的文本格式
        # with open(root_dir+"/data/cutted_sentences.txt","a",encoding="utf8") as fw:
        #     fw.write("__lable__"+str(category)+" , "+" ".join(segs)+"\n")

    #
    classifier=fasttext.train_supervised(root_dir+"/data/cutted_sentences.txt")

    result = classifier.test(root_dir+"/data/cutted_sentences.txt")
    
    print_results(*result)
    classifier.save_model(root_dir +"/models/fasttext.ftz")
    # print("Number of examples:",result.nexamples)    #预测错的例子