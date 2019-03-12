
# coding: utf-8

# In[1]:


import datetime
from pathlib import Path

from gensim.models.word2vec import LineSentence, Word2Vec
from tqdm import tqdm


class MyWord2Vec(object):
    """
    架构：skip-gram（慢、对罕见字有利）vs CBOW（快）
    训练算法：分层softmax（对罕见字有利）vs 负采样（对常见词和低纬向量有利）
    负例采样准确率提高，速度会慢，不使用negative sampling的word2vec本身非常快，但是准确性并不高
    欠采样频繁词：可以提高结果的准确性和速度（适用范围1e-3到1e-5）
    文本（window）大小：skip-gram通常在10附近，CBOW通常在5附近
    """

    def __init__(self, corpus=None):
        """
        docs = [['Well', 'done!'],
                ['Good', 'work'],
                ['Great', 'effort'],
                ['nice', 'work'],
                ['Excellent!'],
                ['Weak'],
                ['Poor', 'effort!'],
                ['not', 'good'],
                ['poor', 'work'],
                ['Could', 'have', 'done', 'better.']]
        model = MyWord2Vec(docs)
        model.word2vec()
        
        model.init_sims(replace=True) # 对model进行锁定，并且据说是预载了相似度矩阵能够提高后面的查询速度，但是你的model从此以后就read only了
        """
        self.corpus = corpus
        self.corpus_convert()

    def word2vec(self,
                 vector_size=300,
                 window=5,
                 min_count=1,
                 sg=0,
                 hs=0,
                 negative=5,
                 epochs=10):
        model = Word2Vec(tqdm(self.corpus, desc="Word2Vec Preprocessing"), size=vector_size, window=window, min_count=min_count, sg=sg, hs=hs,
                         negative=negative, iter=epochs, workers=32)
        return model

    def corpus_convert(self):
        if isinstance(self.corpus, str) and Path(self.corpus).is_file():
            self.corpus = LineSentence(self.corpus)

    def model_save(self, model, path=None):
        if path:
            model.save(path)
        else:
            model.save('./%s___%s.model' % (str(datetime.datetime.today())[:22], model.__str__()))


# In[3]:


skip_gram = MyWord2Vec('/DATA/1_DataCache/FinCorpus/all_data_pure.txt').word2vec(min_count=250, sg=1, hs=1, epochs=2**3)


# In[22]:


skip_gram.save('/DATA/1_DataCache/FinCorpus/skip_gram.model')


# In[28]:


skip_gram.wv.most_similar('保险')


# In[30]:


skip_gram = SGHS('/DATA/1_DataCache/FinCorpus/skip_gram.model')


# In[33]:


skip_gram.key_words(['保险'])


# In[34]:


from sklearn.feature_extraction import text


# In[37]:


l = text.TfidfVectorizer()


# In[42]:


skip_gram.model.wv.most_similar('保险')


# In[27]:


skip_gram = SGHS(skip_gram)


# In[ ]:


skip_gram.key_words(ji)


# In[6]:


from pathlib import Path

import gensim
import numpy as np


class SGHS(object):
    def __init__(self, sg_hs_model=None):
        """
        :param sg_hs_model: Skip-Gram + Huffman Softmax
        """
        self.model = sg_hs_model
        self._model_convert()

    def key_words(self, s):
        s = [i for i in s if self.model.wv.__contains__(i)]
        ws = [(i, sum([self._prob(o, i) for o in s])) for i in s]

        return sorted(ws, key=lambda x: x[1])[::-1]

    def _prob(self, oword, iword):
        x = self.model.wv.word_vec(iword)  # 输入词向量
        oword = self.model.wv.vocab[oword]
        d = oword.code  # 该节点的编码（非0即1）
        p = oword.point  # 该节点的Huffman编码路径

        theta = self.model.trainables.syn1[p].T  # size*n: 300*4

        dot = np.dot(x, theta)  # 4*4
        lprob = -sum(np.logaddexp(0, -dot) + d * dot)  # 估算词与词之间的转移概率就可以得到条件概率了

        return lprob

    def _model_convert(self):
        if isinstance(self.model, str) and Path(self.model).is_file():
            self.model = gensim.models.Word2Vec.load(self.model)

