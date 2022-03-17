from bs4 import BeautifulSoup
#from pip import main
import requests
import pandas as pd
import feedparser
import regex as re
import sqlite3
import time

# Variables
name_of_db = 'all_data.db'
CNN_RSS_URL = "http://rss.cnn.com/rss/cnn_latest.rss"


def get_RSS_data (URL):
    """
    Get data from RSS feed with URL as the RSS link and dataFrame where the data is to be saved
    """
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
    full_news = []
    for i in range(num_of_entries):
        Current_news = news_links[i]
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

            full_news = [article_joined.replace(u'\xa0', u' ')]
        
        elif re.match(advertisement_check, Current_news.link):
            pass

        elif re.match(article_check, Current_news.link):
            r = requests.get(Current_news.link)
            soup = BeautifulSoup(r.text, 'html.parser')
            articles = soup.find_all('div', class_ = 'Paragraph__component')
            article_joined = ""
        
            for j in articles:
                article_live = j.text
                article_joined = article_joined + article_live
            full_news = [article_joined.replace(u'\xa0', u' ')]
        else:
            r = requests.get(Current_news.link)
            soup = BeautifulSoup(r.text, 'html.parser')
            regex_exp = re.compile('.*zn zn-body-text zn-body zn--idx-0 zn--ordinary zn-has-multiple-containers.*')
            news = soup.find('section', {"class": regex_exp}).text
            news = news.replace("(CNN)","")
            full_news = [news.lstrip()]
        temp_df = pd.DataFrame({'Date_Published': publish_date, "Title": news_title, "Summary": news_summary, "News_Link": news_link,"Full News":full_news})
        RSS_DF = pd.concat([RSS_DF, temp_df],ignore_index= True, axis = 0)
    return RSS_DF



## Need to run it only once maybe?

def rss_to_db(database_name):
    data_news = get_RSS_data(CNN_RSS_URL)
    # Open Database
    db_connection = sqlite3.connect(database_name)

    # Save table to database
    today_date = time.strftime("%B %d, %Y")
    table_name = "CNN News from " + today_date
    data_news.drop_duplicates(subset=["Title","News_Link"])
    data_news.to_sql(table_name, db_connection, if_exists = 'append', index = False)
    # Close database
    db_connection.close()


if __name__ == '__main__':
    rss_to_db(name_of_db)

# conn = sqlite3.connect('AA_db.sqlite')
# df = pd.read_sql_query("SELECT * from `CNN NEWS FROM March 16, 2022`", conn)
# conn.close()