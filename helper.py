from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import contractions

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

def news_clustering(news_sentences, k_max = 10):
    s = np.array(news_sentences)
    x = model.encode(s)
    sil = []
    
    for k in range(2, k_max+1):
        km = KMeans(n_clusters= k, init = 'random', n_init = 10, max_iter= 300, tol= 1e-04, random_state= 123)
        y_km = km.fit(x)
        label = y_km.labels_
        sil.append(silhouette_score(x, label, metric = 'euclidean'))
    optimal_k = sil.index(max(sil)) + 1
    
    km = KMeans(n_clusters= optimal_k, init = 'random', n_init = 10, max_iter= 300, tol= 1e-04, random_state= 123)

    y_km = km.fit_predict(x)

    return y_km


# data_set =  

def word_frequency(data_set, n, use_tfidf = False):
    """
    word_frequency takes a list of sentences as input and breaks them down to return the top n number of words in
        data_set. If use_tf0df = True then tfid is used instead of just counting frequency of words
    """
    words, word_count = [], []
    if use_tfidf:
        vectorizer = TfidfVectorizer(stop_words='english').fit(data_set)
    else:   
        vectorizer = CountVectorizer(stop_words='english').fit(data_set)
    
    number_of_words = vectorizer.transform(data_set)
    word_agg = number_of_words.sum(axis = 0)
    word_frequency = [(word, word_agg[0,idx]) for word, idx in vectorizer.vocabulary_.items()]
    word_frequency = sorted(word_frequency, key = lambda word_count: word_count[1], reverse = True)

    for i, j in word_frequency[:n]:
        words.append(i), word_count.append(j)
    return words, word_count



def get_word_viz(words, word_count):
    #fig, ax = plt.subplots(figsize=(15, 10))
    sns.set_theme(style="whitegrid")
    sns.set(rc={"figure.figsize":(15, 8)})
    sns.set(font_scale = 2),
    p = sns.barplot(y = words, x=word_count, palette="Blues_d")
    p.set(ylabel = "Words",
     xlabel = "Word frequency",
     title = "Most common words")

#topic_1_news = result[result.topic_cluster == 1]

def get_topic_pie_viz(news_data):
    """
    creates the pie chart for the topics
    """
    data = news_data.groupby(['Topics']).count()['Title']
    number_of_topics = len(data)
    labels = []
    for i in range(1, number_of_topics + 1):
        lbl = "Topic " + str(i) 
        labels.append(lbl)
    colors = sns.color_palette('pastel')[0:number_of_topics]
    
    #create pie chart
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
    #plt.show()
    return plt
    
#get_topic_pie_viz(result)

def get_hist_word_viz(dataset, feature_name):
    fig, ax = plt.subplots(figsize = (15, 8))
    ax.hist(dataset[feature_name], edgecolor="black", color="#69b3a2", alpha=0.3)
    # Add title and axis names
    plt.title('Total Words Per News Article')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')


def news_cleaner(news_string):
    cleaned_words = []
    for word in news_string.split():
        cleaned_words.append(contractions.fix(word))
    cleaned = " ".join(cleaned_words)
    cleaned = re.sub(r"[^A-Za-z\s]+", "", cleaned)
    cleaned = cleaned.replace("CNN News", "")
    cleaned = cleaned.lower()
    cleaned = " ".join(cleaned.split())
    return cleaned




    