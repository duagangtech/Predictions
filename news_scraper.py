from bs4 import BeautifulSoup
import requests
import pandas as pd
import feedparser
import regex as re
import sqlite3
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score


# Variables
name_of_db = 'all_data.db'
CNN_RSS_URL = "http://rss.cnn.com/rss/cnn_latest.rss"
name_of_table = "CNN_News"

# Embedding model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def remove_duplicate_links(link_list, new_url):
    """
    remove_duplicate(old_url, new_url) return True if new_url is already in link_list and False otherwise
    """
    if new_url == []:
        return True
    else:
        for i in link_list:
            if new_url == i:
                return False
        return True


def get_RSS_data (URL):
    """
    Get data from RSS feed with URL as the RSS link and dataFrame where the data is to be saved
    """
    try:
        conn = sqlite3.connect(name_of_db)
        Current_df = pd.read_sql_query("SELECT * from CNN_News", conn)
        old_url = Current_df['News_Link']
        conn.close()
    except:
        old_url = []
        Current_df = pd.DataFrame(columns = ["Date_Published", "Title", "Summary", "News_Link", "Full News"])

    RSS_DF = pd.DataFrame(columns = ["Date_Published", "Title", "Summary", "News_Link", "Full News"])
    NewsFeed = feedparser.parse(URL)
    news_links = NewsFeed.entries
    num_of_entries = len(news_links)
    publish_date = []
    news_title = []
    news_summary = []
    news_link = []
    live_news_check = re.compile('.*live-news.*')
    article_check = re.compile('.*article.*')
    advertisement_check = re.compile('.*cnn-underscored.*')
    business_check = re.compile('.*business.*')
    full_news = []
    for i in range(num_of_entries):
        Current_news = news_links[i]
        if remove_duplicate_links(old_url, Current_news.link):
            #Current_news = news_links[i]
            publish_date = [Current_news.published]
            news_title = [Current_news.title]
            sentence = Current_news.summary
            head, sep, tail = sentence.partition('<div class=')
            news_summary = [head]
            news_link = [Current_news.link]
            # to avoid sending too many requests at once
            time.sleep(5)
            # Scrape based on news structure
            if re.match(live_news_check, Current_news.link):
                r = requests.get(Current_news.link)
                soup = BeautifulSoup(r.text, 'html.parser')
                live_news = soup.find_all('p', class_ = 'sc-gZMcBi render-stellar-contentstyles__Paragraph-sc-9v7nwy-2 dCwndB')
            
                article_joined = ""
        
                for j in live_news:
                    news_live = j.text
                    article_joined = article_joined + news_live
                article_joined = article_joined.replace("(CNN)","")
                full_news = [article_joined.replace(u'\xa0', u' ')]
        
            elif re.match(advertisement_check, Current_news.link):
                publish_date =[]
                news_title = []
                news_summary = []
                news_link = []
                full_news = []
            elif re.match(article_check, Current_news.link):
                r = requests.get(Current_news.link)
                soup = BeautifulSoup(r.text, 'html.parser')
                articles = soup.find_all('div', class_ = 'Paragraph__component')
                article_joined = ""
        
                for j in articles:
                    article_live = j.text
                    article_joined = article_joined + article_live

                article_joined = article_joined.replace("(CNN)","")
                full_news = [article_joined.replace(u'\xa0', u' ')]

            else:
                r = requests.get(Current_news.link)
                soup = BeautifulSoup(r.text, 'html.parser')
                regex_exp = re.compile('.*zn zn-body-text zn-body zn--idx-0 zn--ordinary zn-has-multiple-containers.*')
                news = soup.find('section', {"class": regex_exp}).text
                news = news.replace("(CNN)","")
                news = news.replace("Sign up for CNN's Wonder Theory science newsletter. Explore the universe with news on fascinating discoveries, scientific advancements and more.","")
                news = news.lstrip()
                if re.match(business_check, Current_news.link):
                    head, sep, tail = news.partition('(CNN Business)')
                    full_news = [tail]
                else:
                    news = news.replace("(CNN Business)","")
                    full_news = [news]
            temp_df = pd.DataFrame({'Date_Published': publish_date, "Title": news_title, "Summary": news_summary, "News_Link": news_link,"Full News":full_news})
            RSS_DF = pd.concat([RSS_DF, temp_df],ignore_index= True, axis = 0)
    return RSS_DF, Current_df


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

def news_cleaner(news_string, is_it_for_BERT=False):
    cleaned = re.sub(r"[^A-Za-z\s]+", " ", news_string)
    cleaned = cleaned.replace("CNN News", "")
    cleaned = cleaned.lower()
    cleaned = " ".join(cleaned.split())
    if is_it_for_BERT:
        return cleaned
    else:
        pass


def rss_to_db(database_name):
    # Get new data if available
    data_news, Current_df = get_RSS_data(CNN_RSS_URL)

    # Cleaning
    data_news['Cleaned Full News'] = data_news['Full News'].apply(lambda x: news_cleaner(x, True))

    # Feature Engineering
    data_news['time_posted'] = pd.DatetimeIndex(data_news['Date_Published']).strftime('%H:%M:%S GMT')
    data_news['Date_posted'] = pd.DatetimeIndex(data_news['Date_Published']).strftime('%Y-%m-%d')
    data_news['Length of post'] = data_news['Full News'].apply(lambda x:len(x.split(" ")))

    # Add Theme Column 
    data_news['Themes'] = np.zeros(data_news.shape[0])

    # Add to existing data
    Current_df = pd.concat([Current_df, data_news],ignore_index= True, axis = 0)

    # Cluster the News
    Current_df['Themes'] = news_clustering(Current_df['Cleaned Full News'], k_max = 10)

    # Cluster the News using this only if table is empty
    #data_news['Themes'] = news_clustering(data_news['Cleaned Full News'], k_max = 10)

    # Open Database
    db_connection = sqlite3.connect(database_name)

    # Save table to database
   # table_name = "Hirdyrts"
    data_news.drop_duplicates(subset=["Title","News_Link"])
    data_news = data_news.sort_values(by='Date_Published')
    data_news.to_sql(name_of_table, db_connection, if_exists = 'append', index = False)
    # Close database
    db_connection.close()


if __name__ == '__main__':
    rss_to_db(name_of_db)
