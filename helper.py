from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import pandas as pd

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# news_sentences = df['Full News']
def func (news_sentences):
    vec = CountVectorizer(stop_words='english').fit(news_sentences)
    bag_of_words = vec.transform(news_sentences)
    sum_words = bag_of_words.sum(axis = 0)
    word_freq = [(word, sum_words[0,idx]) for word,idx in vec.vocabulary_.items()]
    word_freq = sorted(word_freq, key=lambda word_count: word_count[1], reverse= True)
    
    return word_freq

def fun_tf (news_sentences):
    news = news_sentences
    tf_vectorizer = TfidfVectorizer(stop_words='english')
    new_matrix = tf_vectorizer.fit_transform(news)
    return new_matrix

def news_clustering(news_sentences, k_max = 5):
    s = np.array(news_sentences)
    x = model.encode(s)
    sil = []
    #k_max = 5

    for k in range(2, k_max+1):
        km = KMeans(n_clusters= k, init = 'random', n_init = 10, max_iter= 300, tol= 1e-04, random_state= 123)
        y_km = km.fit(x)
        label = y_km.labels_
        sil.append(silhouette_score(x, label, metric = 'euclidean'))
    optimal_k = sil.index(max(sil)) + 1
    
    km = KMeans(n_clusters= optimal_k, init = 'random', n_init = 10, max_iter= 300, tol= 1e-04, random_state= 123)

    y_km = km.fit_predict(x)

    # y_km

    result = pd.DataFrame({'News': news_sentences, 'topic_cluster': y_km})
    
    return result


