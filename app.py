from functools import cache
import streamlit as st
import sqlite3
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import pandas as pd

@st.cache
def my_huggingface_model(model_name):
    model = SentenceTransformer(model_name)
    return model

@st.cache
def get_data(database_name):
    conn = sqlite3.connect(database_name)
    df = pd.read_sql_query("SELECT * from `CNN NEWS FROM March 17, 2022`", conn)
    conn.close()
    return df

@st.cache
def news_clustering(news_sentences, k_max = 5):
    model = my_huggingface_model('sentence-transformers/all-mpnet-base-v2')
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

if __name__ == "__main__":
    st.title('Daily Dashboard')

    st.write('Welcome to my sentiment analysis app!')

    my_data = get_data('all_data.db')['Full News']

    st.write(my_data)

    my_cluster = news_clustering(my_data, 5)

    st.write(my_cluster)

