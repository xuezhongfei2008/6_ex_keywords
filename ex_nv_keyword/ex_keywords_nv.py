# encoding:utf-8

import jieba.posseg as pseg
import pandas as pd
import jieba
# from simi_tfidf.tfidf_similarity import *
import re

jieba.load_userdict('/opt/algor/gongxf/python3_pj/Robot/original_data/finWind_pos0827.txt')
stop_words_path = "/opt/algor/gongxf/python3_pj/Robot/original_data/stop_words.txt"
def stop_words():
    stopwords = [line.strip() for line in open(stop_words_path, 'r', encoding='utf-8').readlines()]
    return stopwords
stopwords_list=stop_words()
stopwords_list.append(' ')
stopwords_list.append('!')
stopwords_list.append('~')



# QUESTION_PATH = '/opt/gongxf/python3_pj/Robot/Ex_keywords/ques.xlsx'


# 词典没有词性标注
def finword():
    finwords_list = []
    finwords = [line.strip() for line in open("/opt/gongxf/python3_pj/Robot/6_ex_keywords/ex_tfidf_keyword/model/key_finWordDict.txt", 'r', encoding='utf-8').readlines()]
    for j in finwords:
        if len(j) > 0:
            j_list = j.split(' ')
            finwords_list.append(j_list[0])
    return finwords_list

def rxd_apply(doc):
    pattern = ('^.*?[我你他她它您亲].*?')
    return re.compile(pattern).findall(doc)

def test(text):
    # setp3: 使用gensim训练TD-IDF 模型，提取关键词
    corpus0 = []
    s_cut = list(pseg.cut(text))
    print("s_cut",s_cut)
    postag = ['an', 'Ng', 'n', 'nr', 'nt', 'ns', 'nz', 'p', 'vg', 'v', 'vd', 'vn', 'vshi', 'vyou', 'vf', 'vx', 'vi',
              'vl', 'vg']
    for ii in s_cut:
        if (ii.flag in postag) and (ii.word not in stopwords_list) and (not rxd_apply(ii.word)):
            corpus0.append(ii.word)
    print("corpus0",corpus0)

if __name__ == '__main__':
    text="您好，任性付是什么东西啊"
    test(text)
    # # step1: 文件输入
    # q_path = "/opt/gongxf/python3_pj/Robot/generate_data/knowledge0720.csv"
    # user_dict = "/opt/gongxf/python3_pj/Robot/Ex_keywords/ex_keyword/model/finWordDict.txt"
    # stop_words_path = "/opt/gongxf/python3_pj/Robot/original_data/stop_words.txt"
    # model_path = "./model"
    #
    # # 测试输入文本 关键字提取
    # df = pd.read_excel(QUESTION_PATH, sheet_name=0, ignore_index=False)
    # for i in range(len(df['question'])):
    #     text=df['question'][i]
    #     print("text",text)
    #     aa = test(text)

