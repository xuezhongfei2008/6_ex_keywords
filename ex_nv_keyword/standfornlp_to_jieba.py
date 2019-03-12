import pandas as pd
df=pd.read_table('/opt/algor/gongxf/python3_pj/Robot/original_data/finWind_pos.txt',sep=' ',header=None,names=['vocab','stand_pos'])

df['num']=100
stanfordnlp_jieba_mapping=dict()
with open('/opt/algor/gongxf/python3_pj/Robot/original_data/stanfordnlp_jieba_mapping.txt') as f:
    for line in f:
        line_stand=line.strip().split(',')
        stanfordnlp_jieba_mapping[line_stand[0]]=line_stand[1]

def replace_stand_jieba(text):
    return stanfordnlp_jieba_mapping[text]
df['jieba_pos']=df['stand_pos'].apply(replace_stand_jieba)
df_fin=df[['vocab','num','jieba_pos']]
df_fin.to_csv('/opt/algor/gongxf/python3_pj/Robot/original_data/finWind_pos0827.txt', sep=' ', header=False, index=False)