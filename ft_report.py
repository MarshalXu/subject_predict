# -*- coding: utf-8 -*-
'''
# Created on 02-18-22 11:05
# @Filename: ft_report.py
# @Desp: 
# @author: xuchang
'''
from sklearn.metrics import classification_report, accuracy_score
import fasttext.FastText as fasttext
from c_common import loadFile
from tqdm import tqdm
import jieba
import os
root_dir = os.path.dirname(__file__)
stopWordPath = os.path.join(root_dir,"data/stopwords.txt")
stopwords = loadFile(stopWordPath)
jieba.load_userdict(os.path.join(root_dir,"data/custom_dict.txt"))

def report(file='', unmatch_path='',model_path = ""):
    rawlabel = []
    prelabel = []
    lines = loadFile(file)
    classifier = fasttext.load_model(model_path)
    for line in tqdm(lines):
        label, content = line.split('\t', 1)  # only one
        # print(label)
        rawlabel.append(int(label.strip().strip('__label__'))-1)
        # print(type(content))
        # print(content)
        prematrix = classifier.predict(content.strip())
        pre_label = prematrix[0][0]
        prelabel.append(int(pre_label.strip().strip('__label__'))-1)
        if label.strip() != pre_label.strip():
            unmatch = pre_label.strip() + '\t' + '|' + '\t' + line + '\n'
        
    # print(rawlabel)
    # print(prelabel)
    # 这里的标签与自己的要对应
    target_names = ['__label__0', '__label__1']
    print(classification_report(rawlabel, prelabel, target_names=target_names, digits=4))
    print("accuracy\t", accuracy_score(rawlabel, prelabel))

def sentence_cut(sentence:str,stopwords = stopwords):
    """
    根据结巴分词:过滤了长度为1的词以及在停用词表中的词
    """
    segs=jieba.lcut(sentence)    #利用结巴分词进行中文分词
    segs=filter(lambda x:len(x)>1,segs)    #去掉长度小于1的词
    segs=filter(lambda x:x not in stopwords,segs)    #去掉停用词

    return " ".join(segs)


def ft_predict(sentence:str,classifier):
    """
    传入需要预测的
    """
    sentence = sentence_cut(sentence)
    print(sentence)
    res = classifier.predict(sentence)  

    pre_label = res[0][0][-1]
    return pre_label

# main主函数修改如下
if __name__ == '__main__':
    model_name = ["model/data_dim200_lr00.5_iter20.model",
    "model/data_dim200_lr00.05_iter20.model"
    ]
    root_dir = os.path.dirname(__file__)
    model_path = os.path.join(root_dir,model_name[0])
    testData_path = os.path.join(root_dir,"data/ft_train.txt")
    report(testData_path,model_path=model_path)
    # cal_precision_and_recall(test_path)
    # result = classifier.test(test_path)
    # print(result)
