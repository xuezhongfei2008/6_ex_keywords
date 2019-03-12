#encoding:utf-8

import jieba.analyse
import jieba.posseg
import logging
import pandas as pd
from simi_tfidf.tfidf_similarity import *
jieba.load_userdict('../original_data/finWordDict.txt')
QUESTION_PATH = '/opt/gongxf/python3_pj/Robot/original_data/金融测评问题集.xlsx'


def test(text,dictionary,tfidf_model,stop_words):
    # #step1: 使用结巴自带的TD-IDF模型 提取关键词
    # tags=jieba.analyse.extract_tags(sentence=text,topK=3,withWeight='withWeight')
    # # print("jieba TD-IDF model"+str(tags))
    #
    # #step2: 使用结巴自带的 TEXTRANK 模型 提取关键词
    # taggs=jieba.analyse.textrank(sentence=text,topK=3,withWeight='withWeight')
    # # print("jieba TD-TEXTRANK model"+str(tags))

    #setp3: 使用gensim训练TD-IDF 模型，提取关键词
    test_cut_raw_0=jieba.lcut(text)
    corpus0=[]
    for item in test_cut_raw_0:
        if item not in stop_words:
            corpus0.append(item)
    test_corpus_0=dictionary.doc2bow(corpus0)
    # print("test_corpus_0",test_corpus_0)
    try:
        test_corpus_tfidf_0=tfidf_model[test_corpus_0]
        test_corpus_tfidf_0=sorted(test_corpus_tfidf_0,key=lambda item:item[1],reverse=True)
        id2token=dict(zip(dictionary.token2id.values(),dictionary.token2id.keys()))
        result0=[]
        for i in range(len(test_corpus_0)):
            # print("id2token[test_corpus_tfidf_0[i][0]",id2token[test_corpus_tfidf_0[i][0]])
            # print("test_corpus_tfidf_0[i][1]",test_corpus_tfidf_0[i][1])
            # result0.append({id2token[test_corpus_tfidf_0[i][0]]:test_corpus_tfidf_0[i][1]})
            result0.append(id2token[test_corpus_tfidf_0[i][0]])
        print(result0)
        # print("gensim TF-IDF model jieba: "+str(result0))
    except Exception:
        logging.error("error")
if __name__=='__main__':

    # step1: 文件输入
    q_path = '../generate_data/client_question_2017.csv'
    user_dict = "../original_data/finWordDict.txt"
    stop_words_path = "../original_data/stop_words.txt"
    # 存放了client_question_2017.csv的tf-idf模型
    model_path = "./keyWords_Model"

    # step2:加载tfidf模型
    SimQuestion_model = SimQuestion(q_path, user_dict, stop_words_path, model_path)
    dictionary, question_list, corpus,tfidf_model, corpus_simi_matrix=SimQuestion_model.build_tfidf_model()
    stop_words = SimQuestion_model.get_stop_words()
    #测试输入文本 关键字提取
    df = pd.read_excel(QUESTION_PATH, sheet_name=0, ignore_index=False)
    for s in df['question']:
    # text="现在急需用点钱，能不能帮我开通一下任性付"
        text=s
        text_list=jieba.lcut(text)
        # print("text_list",text_list)
        test(text,dictionary,tfidf_model,stop_words)





