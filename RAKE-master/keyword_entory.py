# coding: utf-8

# In[85]:


import re
import pymongo
from tqdm import tqdm
import hashlib
from math import log

db_court = pymongo.MongoClient(host='10.244.2.3', port=27017, connect=False).test.robot
md5 = lambda s: hashlib.md5(s).hexdigest()


def texts():
    texts_set = set()
    for court_case in tqdm(db_court.find(no_cursor_timeout=True).limit(1230080)):
        if md5(court_case['question'].encode('utf-8')) in texts_set:
            pass
        else:
            texts_set.add(md5(court_case['question'].encode('utf-8')))
        for t in re.split('[^\u4e00-\u9fa50-9a-zA-Z]+', court_case['question']):
            if t:
                yield t
    print('最终计算了%s篇文章' % len(texts_set))
    return texts_set


# In[57]:


from collections import defaultdict
import numpy as np

n = 4
min_count = 1
ngrams = defaultdict(int)

for t in texts():
    for i in range(len(t)):
        for j in range(1, n + 1):
            if i + j <= len(t):
                ngrams[t[i:i + j]] += 1

ngrams = {i: j for i, j in ngrams.items() if j >= min_count}
total = 1. * sum([j for i, j in ngrams.items() if len(i) == 1])

# In[59]:


len(ngrams)

# In[60]:


from gensim.models import Word2Vec, FastText
import gensim

fasttest_vec = FastText.load('/opt/gongxf/python3_pj/Robot/2_fasttext2vec/all_session/fasttext_cbow.model')
len(fasttest_vec.wv.vocab.keys())

# In[68]:


# 凝固度 刷选
min_proba = {2: 50, 3: 50, 4: 50}
# min_proba={2:0, 3:1, 4:1}
solidify_ngrams = dict()


def is_keep(s):
    if len(s) > 2:
        score = min([total * ngrams[s] / (ngrams[s[:i + 1]] * ngrams[s[i + 1:]]) for i in range(len(s) - 1)])
    else:
        score = total / ngrams[s]
    return score


for word in fasttest_vec.wv.vocab.keys():
    if word in ngrams:
        proba = is_keep(word)
        fre = ngrams[word]
        solidify_ngrams[word] = [fre, proba]

# In[70]:


len(solidify_ngrams)

# In[66]:


from nltk.probability import FreqDist
import math
import re

# In[87]:


# min_entropy=0.8
# def entropy(alist):
#     f=FreqDist(alist)
#     ent=(-1)*sum([i/len(alist)*math.log(i/len(alist)) for i in f.values()])
#     return ent
# #筛选左右熵
# lr_word=dict()
# entory_ngrams_=[]
# for i in solidify_ngrams.keys():
#     lr_word[i]=[]

# for text in texts():
#     for i in solidify_ngrams.keys():
#         lr1=re.findall('(.)%s(.)'%i,text)
#         if lr1:
#             lr_word[i].extend(lr1)

# for i in solidify_ngrams.keys():
#     left_entropy=entropy([w[0] for w in lr_word[i]])
#     right_entropy=entropy([w[1] for w in lr_word[i]])
#     solidify_ngrams[i].append(left_entropy)
#     solidify_ngrams[i].append(right_entropy)


# In[88]:


min_entropy = 0.8
from multiprocessing import Process, cpu_count, Manager

manager = Manager()


def entropy(alist):
    f = FreqDist(alist)
    ent = (-1) * sum([i / len(alist) * math.log(i / len(alist)) for i in f.values()])
    return ent


# 筛选左右熵
lr_word = manager.dict()
entory_ngrams_ = []
for i in solidify_ngrams.keys():
    lr_word[i] = []

corpus = list(solidify_ngrams.keys())
cate_question_len = len(corpus)  # 该类别长度
multi_num = cpu_count()
batch_size = int(cate_question_len / multi_num)
corpus_multi = []
for i in range(multi_num - 1):
    corpus_multi.append(corpus[i * batch_size:(i + 1) * batch_size])
corpus_multi.append(corpus[(multi_num - 1) * batch_size:])  # 每个进程的数据集


def fun(corpus_multi):
    for text in texts():
        for i in corpus_multi:
            lr1 = re.findall('(.)%s(.)' % i, text)
            if lr1:
                lr_word[i].extend(lr1)
    return lr_word


proc_record = []
manager = Manager()
for i in range(multi_num):
    p = Process(target=fun, args=(corpus_multi[i],))
    p.start()
    proc_record.append(p)
for p in proc_record:
    p.join()

for i in solidify_ngrams.keys():
    left_entropy = entropy([w[0] for w in lr_word[i]])
    right_entropy = entropy([w[1] for w in lr_word[i]])
    solidify_ngrams[i].append(left_entropy)
    solidify_ngrams[i].append(right_entropy)

import pickle

with open('solidify_ngrams', 'wb') as fwrite:
    # 序列化
    pickle.dump(solidify_ngrams, fwrite)
