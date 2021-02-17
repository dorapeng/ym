#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %load LDA.py
#!/usr/bin/env python

# In[ ]:


import gensim
from gensim import corpora
from gensim import models
import pandas as pd  # For data handling
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import tqdm
from wordcloud import WordCloud
from collections import OrderedDict
import pyLDAvis
import pyLDAvis.gensim
import pickle


# In[1]:


def make_dic_corpus(bigrams_ig):
    """
    製作dictionary and corpus
    
    Parameters:
    ----------
    text : bigrams_ig
    
    Returns:
    -------
    dictionary : Gensim dictionary(id2word)
    corpus : Gensim corpus
    """
    id2word = corpora.Dictionary(bigrams_ig)
    print('Total Vocabulary Size:', len(id2word))
    corpus = [id2word.doc2bow(text) for text in bigrams_ig]
    return id2word, corpus

def corpus_freq(corpus):
    dict_corpus = {}
    for i in range(len(corpus)):
        for idx, freq in corpus[i]:
            if id2word[idx] in dict_corpus:
                dict_corpus[id2word[idx]] += freq
            else:
                dict_corpus[id2word[idx]] = freq
    dict_df = pd.DataFrame.from_dict(dict_corpus, orient='index', columns=['freq'])
    dict_df.sort_values('freq',ascending = False).head(100)
    return dict_df


def adjust_dic_corpus(id2word,corpus,a,b):
    """
    調整dictionary and corpus
    
    Parameters:
    ----------
    dictionary : Gensim dictionary(id2word)
    corpus : Gensim corpus
    a : no_below 出現次數低於2次 就忽視
    b : no_above 出現次數大於0.6*document總數 就忽視
    
    Returns:
    -------
    dictionary : Gensim dictionary(id2word)
    corpus : Gensim corpus
    """
    id2word.filter_extremes(no_below= a, no_above= b)
    # id2word.filter_extremes(no_below=2)
    print('Total Vocabulary Size:', len(id2word))
    corpus = [id2word.doc2bow(text) for text in bigrams_ig]
    return id2word, corpus

def lda_model(corpus, id2word, k, a, b, num_w):
    """
    LDA模型
    Parameters:
    ----------
    corpus : Gensim corpus
    dictionary : Gensim dictionary(id2word)
    k : number of topics
    a : alpha 設定的數值 (0.01)
    num_w : number of words (print topic及其字詞的方程式時，字詞的數量)

    Returns:
    -------
    model : ldamodel
    """
    ldamodel = gensim.models.LdaModel(corpus = corpus, 
                                      id2word=id2word,
                                      num_topics= k, 
                                      passes=10,  #passes:訓練期間通過語料庫的次數
                                      chunksize = 100, # Number of documents to be used in each training chunk.
                                      alpha = a, 
                                      eta = b, #symmetric is default
        #                                 minimum_probability = 0.1 #概率低於此閾值的主題將被過濾掉。
                                      per_word_topics=True) #the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count).
    topics = ldamodel.print_topics(num_words= num_w)
    for idx, topic in ldamodel.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
    pickle.dump(corpus,open('corpus.pkl','wb')) #存檔備用
    id2word.save('id2word.gensim')
    ldamodel.save('ldamodel.gensim')  
    return ldamodel
     
def lda_multiple_core(corpus, id2word, k, num_w):
    """
    使用多核心的LDA模型，和普通LDA模型差別，除了多核心外，參數也不同，alpha and eta只能是對稱or非對稱
    Parameters:
    ----------
    corpus : Gensim corpus
    dictionary : Gensim dictionary(id2word)
    k : number of topics
    num_w : number of words (print topic及其字詞的方程式時，字詞的數量)

    Returns:
    -------
    model : ldamultimodel
    """
    ldamultimodel = gensim.models.LdaMulticore(corpus, 
                                              num_topics= k, 
                                              id2word=id2word, 
                                              passes=10,  #passes:訓練期間通過語料庫的次數
                                              workers=4,
                                              chunksize = 100, # Number of documents to be used in each training chunk.
                                              alpha = 'symmetric', #is default
                                              eta = 'symmetric', #is default
                                              per_word_topics=True) #the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count).
    topics = ldamodel.print_topics(num_words=num_w)
    for idx, topic in ldamodel.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
    pickle.dump(corpus,open('corpus.pkl','wb')) #存檔備用
    id2word.save('id2word.gensim')
    ldamultimodel.save('ldamultimodel.gensim')  
    return ldamultimodel

def every_weight(ldamodel):
    data_lda = {i: OrderedDict(ldamodel.show_topic(i,25)) for i in range(5)}
    df_lda = pd.DataFrame(data_lda)
    df_lda = df_lda.fillna(0).T
    print(df_lda.shape)
    df_lda.to_csv('lda_every_term_weight.csv', index=True)
    return df_lda

def heat_map(df_lda):
    g=sns.clustermap(df_lda.corr(), center=0, standard_scale=1, cmap="RdBu", metric='cosine', linewidths=.75, figsize=(15, 15))
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.show()
    
def topic_term30(ldamodel,name):
    # TOPIC * 前 30 term
    num_topics = 5
    topic_words = []
    for i in range(num_topics):
        tt = ldamodel.get_topic_terms(i,30)
        topic_words.append([id2word[pair[0]] for pair in tt])
    topic_term_df = pd.DataFrame(topic_words).T
#     topic_term_df.columns = ['topic 1','topic 2', 'topic 3', 'topic 4', 'topic 5', 'topic 6', 'topic 7',  'topic 8']
    topic_term_df.columns = ['topic 1','topic 2', 'topic 3', 'topic 4', 'topic 5']
    topic_term_df.to_csv(name+'.csv')
    return topic_term_df


def wordcloud(ldamodel,name):
    for t in range(ldamodel.num_topics):
        plt.figure()
        a = ldamodel.show_topic(t, 100)
        d = dict(a)  
        wc = WordCloud(max_font_size=50,background_color="white").fit_words(d)
        plt.imshow(wc,interpolation="bilinear")
        plt.axis("off")
        plt.title("Topic #" + str(t))
        plt.show()
        plt.tight_layout() 
        wc.to_file( name + '_topic #' +str(t) +'.png')
        
def compute_coherence(corpus, id2word):
    """
    Compute c_v coherence 
    Parameters:
    ----------
    ldamodel: lda model (fix alpha and eta)
    corpus : Gensim corpus
    dictionary : Gensim dictionary(id2word)
    
    Returns:
    -------
    model : Coherence values
    """
    ldamodel = gensim.models.LdaModel(corpus=corpus, dictionary=id2word, num_topics=k)
    coherence = CoherenceModel(model=ldamodel,
                              corpus=corpus,
                              dictionary=id2word,
                              texts=trigrams_ig,
                              coherence='c_v')
    return coherence.get_coherence()

def topic_num_cv(a,b,c):
    '''
    Decide which number of topic has the best c_v score
    Parameters:
    ----------
    a: min_topics 
    b: max_topics
    c: step_size 
    (2,20,1)
    

    Returns:
    -------
    Plot: number of topics vs. c_v score
    '''
    def compute_coherence_cv(corpus, dictionary, k):
        """
        Compute c_v coherence 
        Parameters:
        ----------
        corpus : Gensim corpus
        dictionary : Gensim dictionary(id2word)
        k: Numbers of topics

        Returns:
        -------
        model : Coherence values
        """
        ldamodel = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=k, 
                                          random_state=123, chunksize=100, passes=10)
        coherence = CoherenceModel(model=ldamodel,
                                  corpus=corpus,
                                  dictionary=dictionary,
                                  texts = trigrams_ig,
                                  coherence='c_v')
        return coherence.get_coherence()
    coherenceList_Cv = []
    min_topics = a
    max_topics = b
    step_size = c
    numTopicsList = list(np.arange(a, b, c))
    for k in numTopicsList:
        cv = compute_coherence_cv(corpus, id2word, k)
        coherenceList_Cv.append(cv)
    plt.plot(numTopicsList, coherenceList_Cv, 'r--')
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    return coherenceList_Cv

def compute_coherence_values(id2word, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    start : start num of topics
    step : append num

    Returns:
    -------
    coherence_values & numbers of topics & alpha & beta .csv
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaModel(id2word=id2word, corpus=corpus, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, corpus=corpus, topics=num_topics, dictionary=id2word, texts=bigrams_ig, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def compute_cv(corpus, id2word, k, a, b): 
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=k, 
                                       random_state=123,
                                       chunksize=100,
                                       passes=10,
                                       alpha=a,
                                       eta=b)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=trigrams_ig, dictionary=id2word, coherence='c_v')
    return coherence_model_lda.get_coherence()

def compute_coherence_values_detail(a, b, c, corpus, id2word, csv_name):
    """
    Compute detailed c_v coherence for various number of topics and various alpha and eta
    alpha = range(0.01, 1, 0.3) 'symmetric' 'asymmetric'
    beta = range(0.01, 1, 0.3) 'symmetric'
    
    Parameters:
    ----------
    texts : List of input texts
    a : min_topics(2 or 3)
    b : max_topics(11)
    c : step_size(1)
    corpus : Gensim corpus
    dictionary : Gensim dictionary
    csv_name: export name 'lda_tuning_results_nostopw_ner.csv'
    
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    import numpy as np
    import tqdm
    grid = {}
    grid['Validation_Set'] = {}
    # Topics range
    min_topics = a
    max_topics = b
    step_size = c
    topics_range = range(min_topics, max_topics, step_size)
    # Alpha parameter
    alpha =list(np.arange(0.01, 1, 0.3))
    alpha.append('symmetric')
    alpha.append('asymmetric')
    # Beta parameter
    beta = list(np.arange(0.01, 1, 0.3))
    beta.append('symmetric')
    # Validation sets
    num_of_docs = len(corpus)
    corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
                   # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
                   gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)), 
                   corpus]
    corpus_title = ['75% Corpus', '100% Corpus']
    model_results = {'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                    }
    # Can take a long time to run
    if 1 == 1:
        pbar = tqdm.tqdm(total=540)
        # iterate through validation corpuses
        for i in range(len(corpus_sets)):
            # iterate through number of topics
            for k in topics_range:
                # iterate through alpha values
                for a in alpha:
                    # iterare through beta values
                    for b in beta:
                        # get the coherence score for the given parameters
                        cv = compute_cv(corpus=corpus_sets[i], id2word=id2word, 
                                        k=k, a=a, b=b)
                        # Save the model results
                        model_results['Validation_Set'].append(corpus_title[i])
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)
                        pbar.update(1)
        a = pd.DataFrame(model_results).to_csv(csv_name, index=False)
        pbar.close()

def vis_coherence_values(data):
    """
    visualize coherence values output
    
    Parameters:
    ----------
    data: '存的資料夾位置/lda_tuning_results.csv'
    
    Returns:
    -------
    visualization
    """
    coherence_score = pd.read_csv(data) 
    df_cv =coherence_score.iloc[:,:5] 
    ax = plt.gca()
    df_cv.plot(kind='line',x='Topics',y='Coherence',ax=ax , title = 'Coherence score and number of topics (fixing alpha=0.01 and beta= symmetric)')

'''
視覺化LDA

Parameters
    ldamodel:輸入model(bow/bigrams/trigrams/bowtfidf)
    corpus
    dic:dictionary
    name:儲存htmls的名字
'''        
def lda_visulization(ldamodel, corpus, id2word, name):     
    lda_display = pyLDAvis.gensim.prepare(ldamodel,corpus,id2word,sort_topics=False) #visulization
    p = pyLDAvis.gensim.prepare(ldamodel, corpus, id2word,sort_topics=False) #save as html
    pyLDAvis.save_html(p, name+'lda.html')
    return pyLDAvis.display(lda_display)   



if __name__ == "__main__":
    id2word, corpus = make_dic_corpus(trigrams_ig)
    dict_df = corpus_freq(corpus)
    id2word, corpus = adjust_dic_corpus(id2word,corpus,a=5,b=0.5)
    ldamodel = lda_model(corpus, id2word, k=5, a='auto', b='auto', num_w=20)
#     ldamultimodel = lda_multiple_core(corpus, id2word, k, num_w)
#     df_lda = every_weight(ldamodel)
#     heatmap = heat_map(df_lda)
#     topic_term_df = topic_term30(ldamodel,'dis_topic_term30_')
#     wordcloud = wordcloud(ldamodel,'dis_0207')
#     compute_coherence(ldamodel, corpus, id2word, trigrams_ig, c)
    decide_num_topic = topic_num_cv(2,20,1)
#     compute_coherence_values(id2word, corpus, texts, limit, start=2, step=3)
#     cv_a_b_k_detail = compute_coherence_values_detail(3, 20, 1, corpus, id2word, 'lda_tuning_results_nostopw_ner.csv')
#     vis_c_v = vis_coherence_values(data)
#     vis_lda = lda_visulization(ldamodel, corpus, id2word, 'dis_0207')

