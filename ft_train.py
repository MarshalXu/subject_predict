# -*- coding: utf-8 -*-
'''
# Created on 02-17-22 17:02
# @Filename: ft_p.py
# @Desp: 
# @author: xuchang
'''

import os
import numpy as np
from tqdm import tqdm
import fasttext.FastText as fasttext
# 如果是装的是FastText的第三方库改为
# import fastText.FastText as fasttext 

root_dir = os.path.dirname(__file__)


def train_model(ipt=None, opt=None, model='', dim=100, epoch=5, lr=0.1, loss='softmax'):
    # suppress: bool, 科学记数法启用
    # True用固定点打印浮点数符号，当前精度中的数字等于零将打印为零。
    # False用科学记数法；最小数绝对值是<1e-4或比率最大绝对值> 1e3。默认值False
    np.set_printoptions(suppress=True)
    if os.path.isfile(model):
        classifier = fasttext.load_model(model)
    else:
        classifier = fasttext.train_supervised(ipt, label='__label__', dim=dim, epoch=epoch,
                                               lr=lr, wordNgrams=2, loss=loss)
        """
          训练一个监督模型, 返回一个模型对象

          @param input:           训练数据文件路径
          @param lr:              学习率
          @param dim:             向量维度
          @param ws:              cbow模型时使用
          @param epoch:           次数
          @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
          @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
          @param minn:            构造subword时最小char个数
          @param maxn:            构造subword时最大char个数
          @param neg:             负采样
          @param wordNgrams:      n-gram个数
          @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
          @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
          @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
          @param lrUpdateRate:    学习率更新
          @param t:               负采样阈值
          @param label:           类别前缀
          @param verbose:         ??
          @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
          @return model object
        """
        classifier.save_model(opt)
    return classifier

if __name__ == '__main__':
    dim = 200
    lr = 5e-1
    epoch = 20
    # 模型存储路径
    # f'string' 相当于 format() 函数
    model = os.path.join(root_dir,f'model/data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model') 
    train_path = os.path.join(root_dir,'data/ft_train.txt') 
    test_path = os.path.join(root_dir,'data/ft_test.txt')
    # 输出原始标签与模型标签不匹配的文本
    unmatch_path = f'unmatch_classification/unmatch_classification_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.txt'
	# 模型训练
    classifier = train_model(ipt=train_path,
                             opt=model,
                             model=model,
                             dim=dim, epoch=epoch, lr=lr#0.5
                             )
    # 模型测试
    result = classifier.test(test_path)
    # 输出结果(测试数据量，precision，recall)：
    print(result)   
    text = [
        "660 扶贫小额贷款 享受 基准利率 抵押 担保 贴息 政策 涉及 贷款 1413.15 万元",
        "认真 整改 以前年度 审计发现的问题",
        "存在 一些 应当 加以 纠正 改进 问题"
    ]
    pre_res = classifier.predict(text[-1])
    print("predict res: ",pre_res)
    