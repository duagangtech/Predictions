# from email import message
from bs4 import BeautifulSoup
from pip import main
import requests
import pandas as pd
import feedparser
import regex as re
import sqlite3
import time
from datetime import datetime

# Variables
name_of_db = 'collect_news.db'
CNN_RSS_URL = "http://rss.cnn.com/rss/cnn_latest.rss"
name_of_table = "CNN_News"
temp_table = "temp_news"

# time now
time_now = datetime.now()
scrape_time = time_now.strftime("%Y-%m-%d") + " " + time_now.strftime("%H:%M")

def get_RSS_data (URL):
    """
    Get data from RSS feed with URL as the RSS link and dataFrame where the data is to be saved
    """
    try:
        conn = sqlite3.connect(name_of_db)
        Current_df = pd.read_sql_query("SELECT * from CNN_News", conn)
        old_new_news = pd.read_sql_query("SELECT * from temp_news", conn)
        # full_news = pd.concat([Current_df, old_new_news],ignore_index= True, axis = 0)
        old_new_news.to_sql(name_of_table, conn, if_exists = 'append', index = False)
        conn.close()
        #old_url = Current_df['News_Link']
    except:
        pass
        #ld_url = []

    RSS_DF = pd.DataFrame(columns = ["Date_Published", "Title", "Summary", "News_Link", "Full News"])
    NewsFeed = feedparser.parse(URL)
    news_links = NewsFeed.entries
    num_of_entries = len(news_links)
    publish_date = []
    news_title = []
    news_summary = []
    news_link = []
    live_news_check = re.compile('.*live-news.*')
    article_check = re.compile('.*/article/.*')
    style_check = re.compile('.*style/article.*')
    advertisement_check = re.compile('.*cnn-underscored.*')
    #business_check = re.compile('.*business.*')
    full_news = []
    for i in range(num_of_entries):
        Current_news = news_links[i]
        #if remove_duplicate_links(old_url, Current_news.link):
        if re.match(advertisement_check, Current_news.link):
            pass
        else:
            publish_date.append(Current_news.published)
            news_title.append(Current_news.title)
            sentence = Current_news.summary
            head, sep, tail = sentence.partition('<div class=')
            news_summary.append(head)
            news_link.append(Current_news.link)
                
            # to avoid sending too many requests at once
            time.sleep(2)
                
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
                full_news.append(article_joined.replace(u'\xa0', u' '))
        
            elif re.match(article_check, Current_news.link):
                r = requests.get(Current_news.link)
                soup = BeautifulSoup(r.text, 'html.parser')
                articles = soup.find_all('div', class_ = 'Paragraph__component')
                article_joined = ""
        
                for j in articles:
                    article_live = j.text
                    article_joined = article_joined + article_live

                article_joined = article_joined.replace("(CNN)","")
                full_news.append(article_joined.replace(u'\xa0', u' '))

            else:
                #print(Current_news.link)
                try:
                    r = requests.get(Current_news.link)
                    soup = BeautifulSoup(r.text, 'html.parser')
                    regex_exp = re.compile('.*zn zn-body-text zn-body zn--idx-0 zn--ordinary.*')
                    news = soup.find('section', {"class": regex_exp}).text
                    news = news.replace("(CNN)","")
                    news = news.replace("Sign up for CNN's Wonder Theory science newsletter. Explore the universe with news on fascinating discoveries, scientific advancements and more.","")
                    news = news.lstrip()
                    news = news.replace("(CNN Business)","")
                    full_news.append(news.replace(u'\xa0', u' '))
                except:
                    publish_date.pop()
                    news_title.pop()
                    news_summary.pop()
                    news_link.pop()
    RSS_DF = pd.DataFrame({'Date_Published': publish_date, "Title": news_title, "Summary": news_summary, "News_Link": news_link,"Full News":full_news})
    return RSS_DF


def rss_to_db(database_name):
    # Get new data if available
    data_news = get_RSS_data(CNN_RSS_URL)

    # Feature Engineering
    data_news['time_posted'] = pd.DatetimeIndex(data_news['Date_Published']).strftime('%H:%M:%S GMT')
    data_news['Date_posted'] = pd.DatetimeIndex(data_news['Date_Published']).strftime('%Y-%m-%d')
    data_news['Length of post'] = data_news['Full News'].apply(lambda x:len(x.split(" ")))
    
    # Open Database
    db_connection = sqlite3.connect(database_name)

    # Save table to database
    data_news.sort_values(by = 'Date_Published', inplace = True)
    #data_news.to_sql(name_of_table, db_connection, if_exists = 'append', index = False)

    result = data_news['Length of post'] == 1
    data_news.drop(data_news[result].index, inplace=True)

    data_news.to_sql(temp_table, db_connection, if_exists = 'replace', index = False)

    message = str(data_news.shape[0]) + " News Articles Added to " + temp_table

    # Keep track of Changes
    #try:
    change_log = pd.DataFrame({'Date': scrape_time, "Title": [message]})
    change_log.to_sql('Change_log', db_connection, if_exists = 'append', index = False)

    # Keep the most recent updated version
    Current_df = pd.read_sql_query("SELECT * from CNN_News", db_connection)
    
    duplicate_rem = Current_df.sort_values('Length of post').drop_duplicates(subset = ['News_Link'], keep="first")

    # Also remove articles that have no content
    result = duplicate_rem['Length of post'] == 1
    duplicate_rem.drop(duplicate_rem[result].index, inplace=True)

    duplicate_rem.to_sql(name_of_table, db_connection, if_exists = 'replace', index = False)

    # Close database
    db_connection.close()


if __name__ == '__main__':
    rss_to_db(name_of_db)
