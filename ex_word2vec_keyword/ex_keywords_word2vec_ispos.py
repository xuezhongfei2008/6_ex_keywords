#encoding:utf-8

import numpy as np
import pandas as pd  #引入它主要是为了更好的显示效果
import jieba
from collections import Counter
from gensim.models import Word2Vec
import jieba.posseg as pseg
import pandas as df

jieba.load_userdict("/opt/gongxf/python3_pj/Robot/original_data/finWind_pos0827.txt")
# model = Word2Vec.load('/DATA/1_DataCache/FinCorpus/skip_gram.model')
# model=Word2Vec.load('./model/cbow_dia.model')
model=Word2Vec.load('/opt/gongxf/python3_pj/Robot/1_word2vec/knowledge0720_cut/knowledge0720_cut.model')


#词典有词性标注的
def finword():
    finwords_list=[]
    finwords_dict={}
    postag = ['an','Ng','n', 'nr', 'nt', 'ns', 'nz','p','vg', 'v', 'vd', 'vn', 'vshi', 'vyou', 'vf', 'vx', 'vi', 'vl', 'vg']
    finwords = [line.strip() for line in open("/opt/gongxf/python3_pj/Robot/6_ex_keywords/ex_tfidf_keyword/model/key_finWordDict.txt",'r', encoding='utf-8').readlines()]
    for j in finwords:
        if len(j)>0:
            j_list=j.split(' ')
            if j_list[2] in postag:                   #提取名词,动词
                finwords_list.append(j_list[0])
                finwords_dict[j_list[0]]=j_list[1]
    # print("finwords_list1:",finwords_list)
    return finwords_list,finwords_dict

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
    # print("lprob",lprob)
    return lprob

def keywords(s):
    s_cut = list(pseg.cut(s))
    print("s_cut",s_cut)
    finwords_list,finwords_dict=finword()
    print("finwords_list",finwords_dict)
    s=[w.word for w in s_cut if w.word in model]
    ws = {w:sum([predict_proba(u, w) for u in s]) for w in s}
    for ii in s_cut:
        # if ii.word in finwords_list and ii.flag == 'n':
        if ii.word in finwords_list:
            print("词性判断:", ii.word, ii.flag)
            ws[ii.word] = float(finwords_dict[ii.word])
    return dict(Counter(ws).most_common(10))


# def test_question():
#     finwords_list,finwords_dict=finword()
#     df = pd.read_excel(QUESTION_PATH, sheet_name=0, ignore_index=False)
#     for s in df['question']:
#         # s_cut = list(pseg.cut(s))
#         key_count = keywords(s)
#         key_word=dict(key_count).keys()
#         print("问题-关键词: ",s,"---",key_word)

if __name__=='__main__':
    # test_question()
    s = '您好，我想问一下任性付怎么不能提前还款'
    print("s",s)
    aa=keywords(s)
    print(aa)

