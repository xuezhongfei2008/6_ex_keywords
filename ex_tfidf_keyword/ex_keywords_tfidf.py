# encoding:utf-8

import jieba.analyse
import jieba.posseg
import logging
import jieba.posseg as pseg
import pandas as pd
from simi_tfidf.tfidf_similarity import *

jieba.load_userdict('/opt/algor/gongxf/python3_pj/Robot/6_ex_keywords/ex_tfidf_keyword/data/finWordDict.txt')
stop_words_path = "/opt/algor/gongxf/python3_pj/Robot/original_data/stop_words.txt"
def stop_words():
    stopwords = [line.strip() for line in open(stop_words_path, 'r', encoding='utf-8').readlines()]
    return stopwords
stopwords_list=stop_words()
stopwords_list.append(' ')
stopwords_list.append('!')
stopwords_list.append('~')

QUESTION_PATH = '/opt/algor/gongxf/python3_pj/Robot/Ex_keywords/ques.xlsx'


# 词典没有词性标注
def finword():
    finwords_list = []
    finwords = [line.strip() for line in open("/opt/algor/gongxf/python3_pj/Robot/6_ex_keywords/ex_tfidf_keyword/model/key_finWordDict.txt", 'r', encoding='utf-8').readlines()]
    for j in finwords:
        if len(j) > 0:
            j_list = j.split(' ')
            finwords_list.append(j_list[0])
    return finwords_list


def test(text, dictionary, tfidf_model,Weights=0.8,size=4):
    # setp3: 使用gensim训练TD-IDF 模型，提取关键词
    corpus0 = []
    s_cut = list(pseg.cut(text))
    print("s_cut",s_cut)
    postag = ['a','an','Ng','n','nr','ns','nt','nz','vg', 'v', 'vd', 'vn', 'vshi', 'vyou', 'vf', 'vx', 'vi', 'vl', 'vg', 'x','r','p']
    posnottag = ['Ag', 'ad', 'b', 'c', 'dg', 'd', 'e', 'f', 'g', 'h', 'i', 'al', 'ld', 'eg', 'y', 'w', 'u', 's']
    finwords_list = finword()
    #词性选择
    for ii in s_cut:
        if (ii.flag in postag) and (ii.word not in stopwords_list):
            corpus0.append(ii.word)
    print("corpus0",corpus0)
    test_corpus_0 = dictionary.doc2bow(corpus0)
    print("test_corpus_0",test_corpus_0)
    try:
        id2token = dict(zip(dictionary.token2id.values(), dictionary.token2id.keys()))
        #tf-idf特征
        test_corpus_tfidf_0 = tfidf_model[test_corpus_0]
        print("test_corpus_tfidf_0",test_corpus_tfidf_0)
        for i in range(len(test_corpus_0)):
            test_corpus_tfidf_0[i]=list(test_corpus_tfidf_0[i])
            if id2token[test_corpus_tfidf_0[i][0]] in finwords_list:
                test_corpus_tfidf_0[i][1]=test_corpus_tfidf_0[i][1]+1.0
        test_corpus_tfidf_0 = sorted(test_corpus_tfidf_0, key=lambda item: item[1], reverse=True)
        print("test_corpus_tfidf_02222",test_corpus_tfidf_0)
        result0 = []
        if len(test_corpus_0) <= size:
            for i in range(len(test_corpus_0)):
                result0.append(id2token[test_corpus_tfidf_0[i][0]])
        else:
            for i in range(int(len(test_corpus_0)*Weights)):
                result0.append(id2token[test_corpus_tfidf_0[i][0]])
        print(result0)
    except Exception:
        logging.error("error")
if __name__ == '__main__':
    # step1: 文件输入
    q_path = "/opt/algor/gongxf/python3_pj/Robot/generate_data/knowledge0720.csv"
    user_dict = "/opt/algor/gongxf/python3_pj/Robot/original_data/finWordDict.txt"
    stop_words_path = "/opt/algor/gongxf/python3_pj/Robot/original_data/stop_words.txt"
    model_path = "./model"

    # step2:加载tfidf模型
    SimQuestion_model = SimQuestion(q_path, user_dict, stop_words_path, model_path)
    dictionary, question_list, corpus, tfidf_model, corpus_simi_matrix = SimQuestion_model.build_tfidf_model()
    stop_words = SimQuestion_model.get_stop_words()

    while True:
        text = input("请输入文本：")
        aa = test(text, dictionary, tfidf_model)

