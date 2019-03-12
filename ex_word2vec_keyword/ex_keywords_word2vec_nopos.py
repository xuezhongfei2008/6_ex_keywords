#encoding:utf-8

import numpy as np
import pandas as pd  #引入它主要是为了更好的显示效果
import jieba
from collections import Counter
from gensim.models import Word2Vec
import jieba.posseg as pseg
import pandas as df
from gensim.models import Word2Vec,FastText
import re

jieba.load_userdict("/opt/algor/gongxf/python3_pj/Robot/original_data/finWind_pos0827.txt")
# model = Word2Vec.load('/DATA/1_DataCache/FinCorpus/skip_gram.model')
model=Word2Vec.load('./model/cbow_dia.model')
# model =  FastText.load('/opt/gongxf/python3_pj/Robot/2_fasttext2vec/all_session/fasttext_cbow.model')
# model=Word2Vec.load('/opt/gongxf/python3_pj/Robot/1_word2vec/knowledge0720_cut/knowledge0720_cut.model')
newword_path='/opt/algor/gongxf/python3_pj/nlp_practice/10_information_extraction/1_new_word/2_better_split_newword'

#词典没有词性标注
def finword():
    finwords_dict=dict()
    # finwords = [line.strip() for line in open("./model/finWordDict1.txt",'r', encoding='utf-8').readlines()]
    finwords = [line.strip() for line in open(newword_path+"/result_robot0827.txt",'r', encoding='utf-8').readlines()]
    for j in finwords:
        # print("j",j)
        if len(j)>0:
            j_list=j.split(',')
            # print("aaa",j_list[1])
            finwords_dict[j_list[1]]=j_list[4]
    return finwords_dict

def remove_rr(doc):
    pattern = ('^.*?[我你他她它您亲].*?|^.*?请问.*?|^.*?一下.*?')
    return re.compile(pattern).findall(doc)
#计算两个词的转移概率
def predict_proba(oword, iword):
    # print("oword,iword",oword,iword)
    #计算P(Wk/Wi) Wi设为关键词
    x = model.wv.word_vec(iword)  # 输入Wi词向量
    oword =model.wv.vocab[oword]
    d = oword.code  # 该节点的Huffman编码（非0即1）
    p = oword.point  # 该节点的Huffman编码路径
    theta = model.trainables.syn1[p].T  # 该节点向量size*n: 300*4
    dot = np.dot(x, theta)  # 4*4
    lprob = -sum(np.logaddexp(0, -dot) + d * dot)  # 估算词与词之间的转移概率就可以得到条件概率了
    return lprob

def norm_ws(ws):
    ws_values=[]
    for i in ws.keys():
        ws_values.append(ws[i])
    value_np=np.array(ws_values)
    amin,amax=value_np.min(),value_np.max()
    value=(value_np-amin)/(abs(amax)+abs(amin))
    for i,key in enumerate(ws.keys()):
        ws[key]=value[i]

def keywords(s,Weights=0.87,size=6):
    s_cut = list(pseg.cut(s))
    postag = ['a','ad','an', 'Ng', 'n', 'nr', 'nt', 'ns', 'nz', 'p', 'vg', 'v', 'vd', 'vn', 'vshi', 'vyou', 'vf', 'vx', 'vi','vl', 'vg','r']
    # postag = ['a', 'an', 'Ng', 'n', 'nr', 'ns', 'nt', 'nz', 'vg', 'v', 'vd', 'vn', 'vshi', 'vyou', 'vf', 'vx', 'vi','vl', 'vg', 'x','r','p']
    posnottag = ['Ag', 'ad', 'b', 'c', 'dg', 'd', 'e', 'f', 'g', 'h', 'i', 'al', 'ld', 'eg', 'y', 'w', 'u', 's']
    print("s_cut",s_cut)
    finwords_dict=finword()
    s=[w.word for w in s_cut if (w.flag in postag) and (not remove_rr(w.word)) and (w.word in model)]
    ws = {w:sum([predict_proba(u, w) for u in s]) for w in s}
    norm_ws(ws)
    print("ws",ws)
    for ii in ws.keys():
        if ii in finwords_dict:
            print("ii",ii)
            print("aaaa",finwords_dict[ii])
            # ws[ii] = float(abs(ws[ii])*100)
            ws[ii] = float(ws[ii])+float(finwords_dict[ii])
    print("ws1",ws)
    if len(ws)<=size:
        len_ws=len(ws)
    elif len(ws)<=10:
        len_ws = int(len(ws) * Weights)
    else:
        len_ws=10
    return dict(Counter(ws).most_common(len_ws))


# def test_question():
#     finwords_list,finwords_dict=finword()
#     df = pd.read_excel(QUESTION_PATH, sheet_name=0, ignore_index=False)
#     for s in df['question']:
#         # s_cut = list(pseg.cut(s))
#         key_count = keywords(s)
#         key_word=dict(key_count).keys()
#         print("问题-关键词: ",s,"---",key_word)

if __name__=='__main__':
    # import pandas as pd
    # df = pd.read_excel('./question.xlsx', sheet_name=0, ignore_index=False)
    # for s in range(len(df['question'])):
    #     aa=keywords(df['question'][s])
    #     print(aa)

    while True:
        s=input("请输入文本：")
        aa=keywords(s)
        print("aa",aa)

    # finword()