#!usr/bin/env python
#encoding:utf-8

# import load_data
import os
import jieba
from gensim import corpora,models,similarities
import pickle


Q_PATH="/opt/algor/gongxf/python3_pj/Robot/generate_data/knowledge0720.csv"
USERDICT_PATH="/opt/algor/gongxf/python3_pj/Robot/Ex_keywords/ex_keyword/model/finWordDict.txt"
STOPWPRD_PATH="/opt/algor/gongxf/python3_pj/Robot/original_data/stop_words.txt"
MODEL_PATH="./model"

class SimQuestion():
    def __init__(self,q_path,user_dict,stop_words_path,model_path):
        jieba.load_userdict(user_dict)
        self.q_path = q_path
        self.stop_words_path=stop_words_path
        self.model_path=model_path
        self.stopwords=self.get_stop_words()
    #读取停止词
    def get_stop_words(self):
        stop_words=[]
        with open(self.stop_words_path,'r',encoding="utf-8") as f:
            line=f.readline()
            while line:
                stop_words.append(line[:-1])
                line=f.readline()
        return stop_words

    #读取文档，并提取标准问题
    def read_data(self):
        all_doc_list=[]
        question_list=[]
        with open(self.q_path,'r',encoding='utf-8') as f:
            line=f.readline()
            while line:
                if(len(line))>0:
                    question_list.append(line.split(sep="\t")[0])    #提取标准问题
                    #使用jieba分词
                    raw_words=list(jieba.cut(line,cut_all=False))
                    #去停用词
                    all_doc_list.append([word for word in raw_words if word not in self.stopwords])
                line=f.readline()
        print("read_data:完毕")
        return all_doc_list,question_list

    def build_tfidf_model(self):
        if (os.path.exists(self.model_path+"/all_doc.dict")):
            # print("加载存在的数据：字典、问题列表、corpus词频数据.....")
            #加载字典
            dictionary = corpora.Dictionary.load(self.model_path+'/all_doc.dict')
            #加载问题列表
            pickle_file=open(self.model_path+'/question_list.pkl','rb')
            question_list=pickle.load(pickle_file)
            #加载corpus 词频数据
            corpus = corpora.MmCorpus(self.model_path+'/all_doc.mm')
        else:
            print("第一次运行生产数据：字典、问题列表、corpus词频数据......")
            all_doc_list,question_list=self.read_data()
            #生成字典并保存
            dictionary=corpora.Dictionary(all_doc_list)
            dictionary.save(self.model_path+'/all_doc.dict')
            #保存问题列表
            pickle_file=open(self.model_path+'/question_list.pkl','wb')
            pickle.dump(question_list,pickle_file)
            pickle_file.close()
            #生产corpus词频数据，并保存
            corpus=[dictionary.doc2bow(doc) for doc in all_doc_list]
            corpora.MmCorpus.serialize(self.model_path+'/all_doc.mm',corpus)

        if (os.path.exists(self.model_path+"/tfidf_model")):
            # print("加载已经存在的模型：tfidf_model.....")
            #加载tfidf模型
            tfidf_model = models.TfidfModel.load(self.model_path+'/tfidf_model')

        else:
            print("需要训练新模型，并保存：tfidf_model.....")
            #生产tfidf模型并保存
            tfidf_model=models.TfidfModel(corpus)
            tfidf_model.save(self.model_path+'/tfidf_model')
        #应用corpus_tfidf模型
        corpus_tfidf = tfidf_model[corpus]
        #生产相似性矩阵索引
        corpus_simi_matrix=similarities.SparseMatrixSimilarity(corpus_tfidf,num_features=len(dictionary.keys()))
        return dictionary,question_list,corpus,tfidf_model,corpus_simi_matrix
    #测试函数
    def test_question(self,test_doc):
        #加载模型数据
        dictionary,question_list,corpus,tfidf_model,corpus_simi_matrix=self.build_tfidf_model()
        # for ii in corpus_simi_matrix.index:
        #     print(corpus_simi_matrix[ii])
        #对测试数据生成词频向量
        test_doc_list=[word for word in jieba.cut(test_doc) if word not in self.get_stop_words()]
        test_doc_vec=dictionary.doc2bow(test_doc_list)
        #计算相似性矩阵,计算要比较的文档与语料库中每篇文档的相似度
        # print(type(corpus_simi_matrix))
        test_simi=corpus_simi_matrix[tfidf_model[test_doc_vec]]
        #按照相似度的值排序
        Similarities_SortIndex=sorted(enumerate(test_simi),key=lambda item:-item[1])
        # print("输入测试文本：",test_doc)
        #输出
        if Similarities_SortIndex[0][1]<=0:
            print("抱歉，我还在学习过程中。。。")
        else:
            for dict in Similarities_SortIndex[0:5]:
                pass
                # print("相似问题：",question_list[dict[0]],dict[1],dict[0])
        return question_list,Similarities_SortIndex

if __name__=='__main__':
    #step1:生产问题文件
    KNOWLEDGE_PATH='/opt/algor/gongxf/python3_pj/Robot/generate_data/knowledge0720.csv'
    if (os.path.exists( "/opt/algor/gongxf/python3_pj/Robot/generate_data/knowledge0720.csv")):
        print("存在Qu.txt文件.....")
    else:
        print("生产Qu.txt文件.....")
        load_data.load_robotknowledge(KNOWLEDGE_PATH)
    #step2:生产tfidf模型
    simquestion_model=SimQuestion(Q_PATH,USERDICT_PATH,STOPWPRD_PATH,MODEL_PATH)
    #step3 测试模型
    test_doc="你好，我想开通任性付，上面的号码还是以前的号码，怎样改成现在的绑定号码？"
    simquestion_model.test_question(test_doc)

