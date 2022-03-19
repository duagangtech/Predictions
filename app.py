from functools import cache
import streamlit as st
import sqlite3
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from helper import *
import math
from wordcloud import WordCloud

# Using plotly.express
import plotly.express as px

st.set_page_config(
     page_title="CNN News Dashboard",
     page_icon= "🐙",
     layout="wide"
 )

# Variables
name_of_db = 'all_data.db'

@st.cache
def get_data(database_name):
    conn = sqlite3.connect(database_name)
    df = pd.read_sql_query("SELECT * from CNN_News", conn)
    conn.close()
    return df


def linePlot(words, word_count):
    fig, ax = plt.subplots(figsize = (20, 6))
    #sns.set_theme(style="whitegrid")
    #sns.set(rc={"figure.figsize":(15, 8)})
    #fig.set(font_scale = 2),
    p = sns.barplot(y = words, x=word_count, palette="Blues_d")
    p.set(ylabel = "Words",
     xlabel = "Word frequency",
     title = "Most common words")

    st.pyplot(fig)

def hist_viz (dataset, feature_name, bins):
    fig, ax = plt.subplots(figsize = (20, 6))
    #plt.rcParams.update({'font.size': 30})
    #plt.rcParams.update({'axes', size': 30}) 
    ax.hist(dataset[feature_name], edgecolor="black", color="#69b3a2", alpha=0.3, bins=bins)
    # Add title and axis names
    plt.title('Total Words Per News Article')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')

    st.pyplot(fig)

def timeseries(df):
    fig = px.line(df, x='date', y="Length of post", title='Length of News article')
    #fig.show()
    #fig.update_layout(xaxis_range=['2022-03-17','2022-03-20'])
    #fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)



## PieChart
def pie_viz(df):
#df = px.data.gapminder().query("year == 2007").query("continent == 'Americas'")
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


words = ["apple", "socks", "toy", "candy"]
word_count = [12,15,1,6]

my_data = get_data('all_data.db')

my_data = my_data.sort_values(by='Date_Published')


## Wordcloud
def wordcloud_viz():
    # Create some sample text
    text = 'Fun, fun, awesome, awesome, tubular, astounding, superb, great, amazing, amazing, amazing, amazing'

    # Create and generate a word cloud image:
    wordcloud = WordCloud(width = 1000, height = 700, colormap="Blues", margin= 2).generate(text)

    # Display the generated image:
    fig, ax = plt.subplots()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    #plt.show()
    st.pyplot(fig)


if __name__ == "__main__":
    st.title('Daily Dashboard')
    
    st.subheader('This is a live breakdown of the CNN news scrpaed from their website using BeautifulSoup and Feedparser. This updates automatically')
    
    st.write("")
    #my_data = get_data('all_data.db')
    
    k = my_data['Themes']
    k = max(k)

    col1, col2, col4 = st.columns(3)

    # Time right now
    now = datetime.now()

    col1.metric("Total News Collected", str(my_data.shape[0]))
    col2.metric("Total Number of Themes/Cluster", str(k))
    #col3.metric("No. of Themes", "9")
    col4.metric("Last Updated", now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S") + " Local Time")
    
    col1, col2 = st.columns([2, 1])

    with col1.container():
        hist_viz (my_data, "Length of post", bins = math.ceil(2*(my_data.shape[0])**(1/3)))

    col2.subheader("Summary")
    col2.write("Here we can see that the longest news has " + "567" + " words")
    col2.write("Here we can see that the shortest news has " + "117" + " words")
    col2.write("The average length of a news is " + "367" + " words" + "with median " + "234" + " words")

    st.write("")

    with st.expander("See explanation"):
     st.write("""
         The chart above shows some numbers I picked for you.
         I rolled actual dice for these, so they're *guaranteed* to
         be random.
     """)
     st.image("https://static.streamlit.io/examples/dice.jpg")


    k = pd.DatetimeIndex(my_data['Date_Published']).strftime('%Y-%m-%d %H:%M:%S')

    my_data['date'] = k
    #timeseries_viz(y, value= my_data["Length of post"])
    timeseries(my_data)
    
    
    st.subheader("This is new section ENTER FILTER HERE")   
    
    col_hist, col_cloud = st.columns([1, 1])

    with col_hist:
        linePlot(words, word_count)

    with col_cloud:
        wordcloud_viz()

    st.write('Welcome to my sentiment analysis app!')


    #wordcloud_viz()
    col1, col2 = st.columns([1, 1])

    with col1:
        pie_viz(my_data)


    col2.subheader("Summary")
    col2.write("Here we can see that the longest news has " + "567" + " words")
    col2.write("Here we can see that the shortest news has " + "117" + " words")
    col2.write("The average length of a news is " + "367" + " words" + "with median " + "234" + " words")


    st.header("Topic Breakdown")

    genre = st.radio(
     "What's your favorite movie genre",
     (['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5']))

    if genre == 'Topic 1':
        st.write('You selected comedy.')
    elif genre == 'Topic 2':
        st.write("You didn't select comedy.")
    else:
        st.write("Here")

    st.dataframe(my_data)

    a = max(my_data['Date_posted'])
    b = min(my_data['Date_posted'])

    st.write (a)

    st.write('Min Date is: ', b)

    