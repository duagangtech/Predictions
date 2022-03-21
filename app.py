from functools import cache
from statistics import median
from tracemalloc import start
from sklearn import cluster
import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from helper import *
import math
from wordcloud import WordCloud
import plotly.express as px


st.set_page_config(
     page_title="CNN News Dashboard",
     page_icon= "🐙",
     layout="wide"
 )

# Variables
name_of_db = 'all_data.db'

def get_data(database_name):
    conn = sqlite3.connect(database_name)
    df = pd.read_sql_query("SELECT * from CNN_News", conn)
    conn.close()
    df = df.sort_values(by='Date_Published')
    df['date'] = pd.DatetimeIndex(df['Date_Published']).strftime('%Y-%m-%d %H:%M:%S')
    return df

@st.cache
def get_metric(dataset):

    #Total Cluster
    number_of_clusters = dataset['Themes']
    number_of_clusters = max(number_of_clusters) + 1

    # Time right now
    now = datetime.now()

    # Total News
    total = my_data.shape[0]

    # get Topics
    labels = []
    for i in range(1, number_of_clusters + 1):
        lbl = "Topic " + str(i) 
        labels.append(lbl)
    
    return number_of_clusters, now, total, labels


@st.cache
def get_news_length_metric(data_set):
    x = data_set['Length of post']
    longest_news = max(x)
    shortest_news = min(x)
    mean_length = round(np.mean(x), 1)
    median_length = round(np.median(x))
    std_length = round(np.std(x), 1)
    return longest_news, shortest_news, mean_length, median_length, std_length

@st.cache(allow_output_mutation=True)
def filter_by_date(data_to_filter, date_input):
    if date_input == 'All':
        return data_to_filter
    else:
        result = my_data['Date_posted'] == date_input
        return data_to_filter[result]


@st.cache
def word_freq(data_set,n, tfidf_use):
    word_frequency(data_set, n, use_tfidf = tfidf_use)

@st.cache
def cleaner_nlp(data_set, feature_name):
    """
    feature_name must be a string and data_set a pandas dataframe
    """
    summary = data_set[feature_name].apply(lambda x: news_cleaner(x))
    return summary

@st.cache(allow_output_mutation=True)
def filter_themes(data_to_filter, user_input):
    result = data_to_filter['Themes'] == user_input
    return data_to_filter[result]

def linePlot(words, word_count):
    fig, ax = plt.subplots(figsize = (15, 10))
    p = sns.barplot(y = words, x=word_count, palette="Blues_d")
    p.set(ylabel = "Words",
     xlabel = "Word frequency",
     title = "Most common words")

    st.pyplot(fig)

def hist_viz (dataset, feature_name, bins):
    fig, ax = plt.subplots(figsize = (20, 6))
    ax.hist(dataset[feature_name], edgecolor="black", color="#6bafd6", bins=bins)
    # Add title and axis names
    plt.title('Total Words Per News Article')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')

    st.pyplot(fig)

def timeseries(df):
    fig = px.line(df, x='date', y="Length of post", title='Length of News article')
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)



## PieChart
def pie_viz(df):
    fig, ax = plt.subplots(figsize = (10, 8))
    data = df.groupby(['Themes']).count()['Title']
    number_of_topics = len(data)
    labels = []
    for i in range(1, number_of_topics + 1):
        lbl = "Topic " + str(i) 
        labels.append(lbl)
    colors = sns.color_palette('pastel')[0:number_of_topics]
    
    #create pie chart
    plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')

    st.pyplot(fig)



## Wordcloud
@st.cache(allow_output_mutation=True)
def wordclouds(words, word_count):
   
    df = pd.DataFrame({'word': words,
                   'count': word_count})
    data = df.set_index('word').to_dict()['count'] 

    # Create and generate a word cloud image:
    wordcloud = WordCloud(width = 600, height = 400, colormap="Blues", margin= 1, max_words=200).generate_from_frequencies(data)
    return wordcloud

def wordcloud_viz(words, word_count):
    # Display the generated image:
    fig, ax = plt.subplots()

    wordcloud = wordclouds(words, word_count)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    #
    st.pyplot(fig)

## Theme Clustering

@st.cache(allow_output_mutation=True)
def cluster_news (dataset, feature_to_cluster):
    """
    dataset -> pandas Dataframe
    feature_to_cluster -> string
    """
    k = 10
    themes = news_clustering(dataset[feature_to_cluster], k_max = k)
    return themes

my_data = get_data('all_data.db')

#my_data['Themes'] = cluster_news(my_data, 'Full News')

@st.cache(allow_output_mutation=True)
def cluster_filter(dataset):
    my_dates = np.array(dataset['Date_posted'].unique())
    date_option = np.append("All", my_dates)
    new_dict = {}
    for i in date_option:
        new_data = filter_by_date(dataset, i)
        theme = cluster_news(new_data, 'Full News')
        new_dict[i] = theme
    return new_dict

with st.spinner('Clustering the news...Almost There!!'):
    # Apply only once
    cluster_dict = cluster_filter(my_data)

# Error tbw
my_data['Themes'] = cluster_dict["All"]

if __name__ == "__main__":
    
    #my_data = get_data('all_data.db')
    # k = pd.DatetimeIndex(my_data['Date_Published']).strftime('%Y-%m-%d %H:%M:%S')
    # my_data['date'] = k

    # Cluster my news into groups

    st.title('Daily Dashboard')
    
    st.subheader('This is a live breakdown of the CNN news scraped from their website using BeautifulSoup and Feedparser. This updates automatically')
    
    st.write("")
    
    #################################################### Top Metric ######################################################
    
    col1, col2, col3 = st.columns(3)

    clusters, time_now, total_news, Topics = get_metric(my_data)

    col1.metric("Total News Collected", str(total_news))
    col2.metric("Total Number of Themes/Cluster", str(clusters))
    col3.metric("Last Updated", time_now.strftime("%Y-%m-%d"), time_now.strftime("%H:%M") + " UTC")
    
    #######################################################################################################################

    # Dataframe/ Data Overview
    st.subheader("Let us begin by looking at the news articles collected so far")
    st.write("The Table below shows the Date, Title and Summary of the 10 latest News collected")

    # CSS to inject contained in a string
    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    st.table(my_data[['Date_Published','Title','Summary']].tail(10))

    with st.expander("Inorder to view all the news Click me!"):
        st.write("Including the full news article made the table hard to read, so I included the news link instead")
        st.dataframe(my_data[['Date_Published','Title','Summary', 'Full News', 'News_Link']])
            

    ## Word Length Viz
    st.header("Now that we see our news, its time to explore!")
    st.subheader("The news articles vary in length, so we will check out the summary of the length of the news articles")
    st.write("The histogram below shows us the distribution of the length of each news article collected so far.")


    st.write("We can filter by dates:")

    ## Filter data by dates
    my_dates = np.array(my_data['Date_posted'].unique())
    date_option = np.append("All", my_dates)

    option = st.selectbox(
     'Choose a date to filter by:',date_option)

    filtered_data_by_date = filter_by_date(my_data, option)
    
    longest,shortest, mean_len, median_len, std_len = get_news_length_metric(filtered_data_by_date)
    
    col1, col2 = st.columns([2, 1])

    with col1.container():
        hist_viz (filtered_data_by_date, "Length of post", bins = math.ceil(2*(total_news)**(1/3)))

    col2.subheader("Summary")
    col2.write("Here we can see that the longest news has " + str(longest) + " words")
    col2.write("Here we can see that the shortest news has " + str(shortest) + " words")
    col2.write("The average length of a news is " + str(mean_len) + " words" + "with median " + str(median_len) + " words and standard deviation of " + str(std_len))

    st.write("")

    st.write("Similarly, we can also see the breakdown of the length of the articles by the time posted")

    # timeseries
    timeseries(my_data)
    
    ## NLP ANALYSIS

    # Tilte and Summary
    st.header("Exploring the Title and Summary of the news articles")

    st.subheader("Title and Summary Analysis")   

    col_hist, col_cloud = st.columns([3, 1])

    with col_cloud:
        option_2 = st.selectbox('Choose a date to filter by:', date_option, key=123)

    title_data_by_date = filter_by_date(my_data, option_2)

    words, word_count = word_frequency(cleaner_nlp(title_data_by_date,'Title'), 15, use_tfidf= False)

    with col_hist:
        st.write("The barplot below shows us the 15 most common words used in all the news title. This gives us a quick overview of what the news is about")
        
        linePlot(words, word_count)

    option_3 = st.selectbox('Choose a date to filter by:', date_option, key=321)
    
    st.write("""A wordcloud is a great way to summarize long texts. Here, I used tfidf to choose the words instead of just counting the
     terms like I did above. This penalizes the common words and helps us find the more relevant words from each document.""")
    
    summary_data_by_date = filter_by_date(my_data, option_3)

    words, word_count = word_frequency(cleaner_nlp(summary_data_by_date,'Summary'), 100, use_tfidf= True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        wordcloud_viz(words, word_count)
    
    
    ########################################## Cluster Documents #####################################################
    st.header("Clustering news that are simialar to each other to get common themes")

    st.subheader("Quick Breakdown of Common themes for All News")

    col1, col2 = st.columns([1, 1])

    with col1:
        pie_viz(my_data)

    with col2:
        st.subheader("Overview")

        st.write("All the news articles can be divided into " + str(clusters) + " clusters")

        st.write("""This was done using SentenceTransformers from the HuggingFace Library that produces BERT embeddings for the sentences from each news article. Once the 
        sentence embeddings were extracted, KMeans Clustering was used to group the similar news articles together. So all the news articles that are similar to each are
        grouped into the same 'Topic'.""")

    st.subheader("We can check out all the news in each group and filter by both the date posted and group")

    option_4 = st.selectbox('Choose a date to filter by:', date_option, key=31)
    
    summary_data_by_date = filter_by_date(my_data, option_4)

    # Cluster my news into groups
    #summary_data_by_date['Themes'] = 
    # cluster_news(summary_data_by_date, 'Full News')

    st.write ("We can check out all the news in each group")

    col1, col2 = st.columns([1, 1])
    summary_data_by_date['Themes'] = cluster_dict[option_4]
    clusters_filtered, time_now_filtered, total_news_filtered, Topics_filtered = get_metric(summary_data_by_date)
    
    with col1:
        theme_choose = st.radio(
     "Check out News for each group",
     Topics_filtered)

        user_input = int(theme_choose[6:]) - 1

        theme_data = filter_themes(summary_data_by_date, user_input)
        try:
            words_themes, word_count_themes = word_frequency(cleaner_nlp(theme_data, "Full News"), 100, use_tfidf= True)
        except:
            # Bug needs to be fixed!
            words_themes, word_count_themes = word_frequency(cleaner_nlp(theme_data, "Summary"), 100, use_tfidf= True)

    longest, shortest, mean_len, median_len, std_len = get_news_length_metric(theme_data)

    col2.subheader("Summary")
    col2.write("Here we can see that the longest news has " + str(longest) + " words")
    col2.write("Here we can see that the shortest news has " + str(shortest) + " words")
    col2.write("The average length of a news is " + str(mean_len) +
     " words" + "with median " + str(median_len) + " words and standard deviation of " + str(std_len))
    col2.write("Total News Articles " + str(theme_data.shape[0]))

    col1_themes, col2_themes = st.columns([1, 1])
    
    with col1_themes.container():
        linePlot(words_themes[:15], word_count_themes[:15])

    with col2_themes:
        wordcloud_viz(words_themes, word_count_themes)

st.subheader("News in each group")

st.dataframe(theme_data[['Title','Summary', 'News_Link']])
st.write("")
st.header("Coming Soon: Sentiment Analysis (Once I have a bit more data!)")
