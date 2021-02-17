#!/usr/bin/env python
# coding: utf-8

# In[3]:


#如果您在研究中使用ScispaCy，請引用《ScispaCy：生物醫學自然語言處理的快速而強大的模型》。另外，請指出您使用的ScispaCy版本和型號，以便可以復制您的研究成果。
#https://github.com/allenai/scispacy
import os
import gensim
from gensim import corpora
from gensim import models
import re  # For preprocessing
import pandas as pd  # For data handling
import spacy  # For preprocessing
import nltk
from nltk.corpus import stopwords
import pdb
import gensim
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
import seaborn as sns
import scispacy      #用於ner                                                                                                              
# from spacy import displacy        #visualizer.                                          
from django.shortcuts import render 
# from scispacy.linking import EntityLinker
from collections import Counter
nlpig = spacy.load('en_core_sci_lg')

get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_colwidth=500 
pd.options.display.max_rows = 500 #設定Output的呈現



'''
1. Import data in csv file format
2. Select discussion and abstract part respectively, lowercase, remove url, and only leave lowercase letters
3. Import custom stopwords
4. Dataframe to list
'''
def load_origine_and_stopw(data):
    def load_original_data(data):
        a = pd.read_csv(data,dtype=str)
        origine = a.iloc[:,:6] #篩選出資料內的某些欄位
        return origine
    origine =  load_original_data(data)   
    def stoplist(stop_list):
        stop = pd.read_csv(stop_list) 
        slist = stop['stop_term'].values.tolist()
        clean = [i.strip() for i in slist] #刪除stoplist中，term的前後空格
        print('自定義的停用詞長度: ', str(len(clean))) #知道長度
        return clean  
    stopw = stoplist(stop_list)
    return origine,stopw

def select_dis_part(origine):
    # Convert lowercase
    origine['Discussion'] = origine['Discussion'].astype(str).map(lambda x: x.lower())
    # Remove punctuation
    origine['Discussion'] = origine['Discussion'].map(lambda x: re.sub(r'([^a-z])', ' ', x))
    dis_df = origine['Discussion']
    dis_list = dis_df.values.tolist()
    return dis_df, dis_list


def select_abst_part(origine):
    origine['Abstract'] = origine['Abstract'].astype(str).map(lambda x: x.lower())
    origine['Abstract'] = origine['Abstract'].map(lambda x: re.sub(r'([^a-z])', ' ', x))
    abst_df =   origine['Abstract'] 
    abst_list = abst_df.values.tolist()
    return abst_df, abst_list



'''
前處理
1. 刪除字元長度<2的字詞 (revised cause: na)
2. 詞型還原，解決單複數問題，(or plus 只留下詞性為名詞、動詞、形容詞、副詞的字詞)
3. 移除停用詞
5. 將每個檔案的名詞子句整合，進行large model(en_core_web_ig)的NER，之後做兩件事: (1)形成document (2)形成document的token模式

'''
def preprocessing(dis_list):
    
    def sent_to_words(dis_list): #如果字長度<3就刪掉
        for document in dis_list:
            yield(gensim.utils.simple_preprocess(str(document), min_len=2 ))#, deacc=True))  # deacc=True removes punctuations
            #default : gensim.utils.simple_preprocess(doc, deacc=False, min_len=2, max_len=15)
    data_words = list(sent_to_words(dis_list)) #by documents 
    print('1. 刪除字元長度<3後的 第一篇文章前30個字:\n',data_words[:1][0][:30],'\n')
    
    # 轉成單數 詞性
    def lemmatization(data_words):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in data_words:
            doc = nlpig(' '.join(sent)) 
            texts_out.append([token.lemma_ for token in doc])    
        return texts_out
    data_lemmatized = lemmatization(data_words)
    data = [' '.join(doc) for doc in data_lemmatized]
    data_lemmatized = [data[i].replace('-PRON-','') for i in range(len(data))]

    
#     def nvaalemma(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#         """https://spacy.io/api/annotation"""
#         texts_out = []
#         for sent in data_words:
#             doc = nlpig(" ".join(sent)) 
#             texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags or token])
#         return texts_out
#     allow_pos = nvaalemma(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#     allow_pos = nvaalemma(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#     data = [' '.join(doc) for doc in allow_pos]
#     allow_pos = [data[i].replace('-PRON-','') for i in range(len(data))]
    
#     print('2. 接續上步驟，lemma後，只留下名詞、動詞、形容詞、副詞的 第一篇文章前30個字:\n', allow_pos[:1][0][:30],'\n')
    
    
    def remove_stopwords(data_lemmatized):
        stopwords = set(nltk.corpus.stopwords.words('english'))
        stopwords.update(stopw)
        return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in data_lemmatized]
    data_words_nostops = remove_stopwords(data_lemmatized)
#     data_words_nostops = remove_stopwords(allow_pos) #if nvaa 
    print('3. 接續上步驟，移除停用詞後的 第一篇文章前30個字:\n',data_words_nostops[:1][0][:30],'\n')

    def doc_ent_and_token(data_words_nostops):
        doc_ig_ent_list = []
        doc_ig_ent_token_list = []
        doc_cd_ent_list = []
        doc_cd_ent_token_list = []
        data = [' '.join(doc) for doc in data_words_nostops]
        for doc in data:
            docig =  nlpig(doc) #large model
            entity_ig_list = [entity for entity in docig.ents]
            a = ' '.join(str(e) for e in entity_ig_list)
            token = a.split()
            doc_ig_ent_list.append(a)
            doc_ig_ent_token_list.append(token)
            
        return doc_ig_ent_list, doc_ig_ent_token_list, doc_cd_ent_list, doc_cd_ent_token_list
    doc_ig_ent_list,doc_ig_ent_token_list, doc_cd_ent_list, doc_cd_ent_token_list  = doc_ent_and_token(data_words_nostops)       
    print('4. 接續上步驟，進行large model(en_core_web_ig)的NER，將NER整合後，其中第一篇文章的結果:\n', doc_ig_ent_list[0], '\n' )
    print('5. 將該結果變成Token模式，其中第一篇文章的結果:\n', doc_ig_ent_token_list[0], '\n' )
    return doc_ig_ent_list, doc_ig_ent_token_list

'''
gensim.models.Phrases
Parameter
    sent（可迭代的str的列表，可選）–可迭代的句子可以只是一個列表, token格式
    min_count（float ，可選）–忽略總收集計數低於此值的所有單詞和雙字母組。 5:至少要一起出現五遍
    threshold（float ，可選）-表示形成短語的分數閾值（越高意味著短語越少）
    max_vocab_size（int ，可選）–詞彙表的最大大小（令牌數）。用於控制不太常用的單詞的修剪，以控制內存
'''
def make_ngram_mod(doc_ig_ent_token_list, num_count):
    bigram = gensim.models.Phrases(doc_ig_ent_token_list, min_count=num_count, threshold=1)
    trigram = gensim.models.Phrases(bigram[doc_ig_ent_token_list], min_count=5, threshold=1)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return bigram_mod,trigram_mod

def make_ngram(doc_ig_ent_token_list):
    bigrams = [bigram_mod[doc] for doc in doc_ig_ent_token_list]
    trigrams=  [trigram_mod[bigram_mod[doc]] for doc in doc_ig_ent_token_list]
    return bigrams,trigrams

def description(origine):
    def vis_j(origine):
        order = origine.Journal.value_counts(ascending=True).index
        plt.figure(figsize=(20,20),dpi=600,linewidth = 0.6)
        plt.figure(figsize=(20,20))
        plt.title("Journals with sglt2-i related articles from 2015 to 2020")
        sns.set(font_scale=2)
        a = sns.countplot(y='Journal', data=origine, orient='h', order=order)
 
        plt.figure(figsize=(5,5),dpi=400,linewidth = 0.6)
        plt.title("Top 10 journals that frequently publish sglt2-i related documents from 2015 to 2020")
        sns.set(font_scale=2)
        b = origine['Journal'].value_counts(ascending=False)[:10].plot( kind='bar')

        plt.figure(figsize=(5,5),dpi=400,linewidth = 0.6)
        plt.title("Least 10 journals that frequently publish sglt2-i related documents from 2015 to 2020")
        sns.set(font_scale=2)
        c = origine['Journal'].value_counts(ascending=True)[:10].plot( kind='bar')

        return a,b,c
    all_vis_j,top_10_vis_j, least_10_vis_j = vis_j(origine)
    
    def vis_year (origine):
        sns.set(font_scale=2)
        plt.figure(figsize=(10,6))
        plt.title("The number of sglt2-i related documents published annually from 2015 to 2020")
        a = sns.countplot(x='Published_year', data=origine) 
        return a
    vis_year = vis_year (origine)
    
#     def journal_year(origine):
#         b = pd.crosstab(origine.Journal, origine.Published_year)
#         return b
#     journal_count = journal_year(origine)
    
    def vis_journal_count(origine):
        sns.set(font_scale=10)
        a = sns.catplot(data=origine, x='Published_year', col='Journal', aspect =3 ,col_wrap=3, sharex=False, height=15,kind='count')
        a.fig.suptitle("The number of sglt2-i related documents in each journal from 2015 to 2020",
                       fontsize = 'x-large' ,  
                       fontweight = 'bold' ) 
        a.fig.subplots_adjust( top = 0.85 ) 
        return a
    journal_year =  vis_journal_count(origine)   
                          
    return all_vis_j,top_10_vis_j, least_10_vis_j, vis_year ,journal_year

                          
                          
if __name__ == "__main__":
    data_dis = 'C:/Users/user/Desktop/Graduate/Data/Discussion_36.csv'
    data_abst = 'C:/Users/user/Desktop/Graduate/Data/Abstract_227.csv'
    stop_list = 'C:/Users/user/Desktop/Graduate/Data/stopw_0213.csv' 
    
#     origine,stopw = load_origine_and_stopw(data_dis)
#     dis_df, dis_list = select_dis_part(origine)
#     doc_ig_ent_list, doc_ig_ent_token_list = preprocessing(dis_list)
    
    origine,stopw = load_origine_and_stopw(data_abst)
    abst_df, abst_list = select_abst_part(origine)
    doc_ig_ent_list, doc_ig_ent_token_list = preprocessing(abst_list)
    
    bigram_mod,trigram_mod = make_ngram_mod(doc_ig_ent_token_list, 5)
    bigrams_ig,trigrams_ig = make_ngram(doc_ig_ent_token_list)
    print('第一篇文章的bigrams結果: \n', bigrams_ig[0:1] , '\n')
    print('第一篇文章的trigrams結果: \n',trigrams_ig[0:1], '\n') 

    all_vis_j,top_10_vis_j, least_10_vis_j, vis_year, journal_year = description(origine)

