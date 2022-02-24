import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from DataExtractor import *
from subprocess import call,run
import isodate

st.set_page_config(
     page_title="Youtube Dashboard",
     page_icon= "🐙",
     layout="wide"
 )

st.write("""
# Youtube Realtime Dashboard

The default search term is set to Youtube. Change it to any term you would like to search for. This will provide a live dashboard of the top 50 relevant search results.
"""
)

search_query = st.text_input(label = "", value="", max_chars=None, key="", type="default", placeholder ="Search term", help="Enter word or phrase to search")

# Search and Reset Buttons
col1, col2,col3 = st.columns([1,1,6])
search_button = col1.button('Click Me to Search!')
reset_button = col2.button('Click Me to Reset')

# Add a space after the buttons
st.text("")
st.text("")

# empty container which will contain our visualizations later once user clicks on search
result_display = st.empty()


# Cards at the top 
metric1, metric2 = st.columns(2)
metric3, metric4 = st.columns(2)


# Layout for first header
header1 = st.empty()
# Layout for the first dataframe
col1, col2, col3 = st.columns([1, 2, 1])

# Layout for second header
header2 = st.empty()
# Layout for the first dataframe
stat1, stat2, stat3 = st.columns([1, 5, 1])

# Full Table Display container
display_full = st.empty()

# container for first scatter plot
date_scatter_heading = st.empty()

# container for first scatter plot
date_scatter_filter = st.empty()

# container for first scatter plot
date_scatter = st.empty()

# Bottom 2 scatter plots
left_plot = st.empty()

right_plot = st.empty()

# Histogram container
Hist = st.empty()

# Corr plot container
Correlation_container = st.empty()

# When rest button is clicked
if reset_button:
    result_display.empty()
    header1.empty()
    header2.empty()
    metric1.empty()
    metric2.empty() 
    metric3.empty() 
    metric4.empty()
    left_plot.empty()
    right_plot.empty()
    Hist.empty()
    Correlation_container.empty()

# When user clicks on search button Replace the empty container with the result of the search:


# When search button is clicked
if search_button:
    if search_query != "":
        # Call main function
        call(['py', 'DataExtractor.py', search_query])
        # read in updated data
        music_data = pd.read_csv('music.csv')
    
        # Get Most Liked Video
        like_max = music_data["Total Likes"].idxmax()
        like_name = music_data.iloc[like_max,0]
        like_value = music_data.iloc[like_max,5]

        # Get Most Viewed Video
        v_max = music_data["Total Views"].idxmax()
        v_name = music_data.iloc[v_max,0]
        v_value = music_data.iloc[v_max,4]

        # Get Most Commented Video
        c_max = music_data["Total Likes"].idxmax()
        c_name = music_data.iloc[c_max,0]
        c_value = music_data.iloc[c_max,6]

        # Get Longest Video
        l_max = music_data["Duration (in minutes)"].idxmax()
        l_name = music_data.iloc[l_max,0]
        l_value = music_data.iloc[l_max,3]

        # Get Shortest Video
        s_min = music_data["Total Likes"].idxmin()
        s_name = music_data.iloc[s_min,0]
        s_value = music_data.iloc[s_min,3]

        # Display all output
        with result_display.container():
            st.metric("Most Liked Video", like_name, str(like_value), delta_color="off")
            metric1.metric("Most Viewed Video", v_name, str(v_value), delta_color="off")
            metric2.metric("Most Commented Video", c_name, str(c_value), delta_color="off")
            metric3.metric("Longest Video", l_name, str(l_value) + " minutes", delta_color="off")
            metric4.metric("Shortest Video", s_name, str(s_value) + " minutes", delta_color="off")
            header1.write("""
            # Search Result 
            """
            )
            name_df = music_data.loc[:,['title', 'Channel Name', 'Release Date (Day-Month-Year)']]
            col2.dataframe(name_df)
            header2.write("""
            # Videos Statistics 
            """)
            stats_df = music_data.loc[:,['title','Duration (in minutes)','Total Views', 'Total Likes', 'Total Comments']]
            stat2.dataframe(stats_df)
            with display_full.container():
                with st.expander("Click Me to View all the Videos"):
                    st.table(music_data)
        with date_scatter_heading:
            st.write("""
            # Video Feature Explorations
                    """
                    )
        with date_scatter:
            fig = px.scatter(music_data, x = "Release Date (Day-Month-Year)", y= "Total Views", size = music_data["Duration (in minutes)"],
            color = "title", title="Total Views by Date of Release", template = 'simple_white')
            date_scatter.plotly_chart(fig, use_container_width=True)
        with left_plot:
            st.write("### Total Comments Vs Total Likes")
            fig = px.scatter(music_data, x = "Total Comments", y="Total Likes", size = music_data["Duration (in minutes)"],\
                color = "title", template = 'simple_white', title="Total Comments Vs Total Likes")
            st.plotly_chart(fig, use_container_width=True)

        with right_plot:
            fig = px.scatter(music_data, x = "Duration (in minutes)", y="Total Views", color = "title", template = 'simple_white',
            title="Total Views Vs Duration")
            st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with Hist:
            with left:
                st.write("### Histogram of Likes")
                fig = px.histogram(music_data, x= music_data["Total Likes"], text_auto=True, opacity=0.8, marginal="box",
                template = 'simple_white', color_discrete_sequence = ['#316395'])

                fig.update_layout(yaxis_title_text='Frequency', xaxis_title_text='Total Likes',
                bargap=0.01, hoverlabel=dict(bgcolor="white", font_size=20))
                st.plotly_chart(fig, use_container_width=True)
        with Hist:
            with right:
                st.write("### Histogram of Views")
                fig = px.histogram(music_data, x= music_data["Total Views"], opacity=0.8, marginal="box",
                template = 'simple_white', color_discrete_sequence = ['#316395'])

                fig.update_layout(yaxis_title_text='Frequency', xaxis_title_text='Total Views',
                bargap=0.01, hoverlabel=dict(bgcolor="white", font_size=20))
                st.plotly_chart(fig, use_container_width=True)

        left1, right1 = st.columns(2)
        with Hist:
            with left1:
                st.write("### Histogram of Comments")
                fig = px.histogram(music_data, x= music_data["Total Comments"], text_auto=True, opacity=0.8, marginal="box",
                template = 'simple_white', color_discrete_sequence = ['#316395'])

                fig.update_layout(yaxis_title_text='Frequency', xaxis_title_text='Total Comments',
                bargap=0.01, hoverlabel=dict(bgcolor="white", font_size=20))
                st.plotly_chart(fig, use_container_width=True)

        with Correlation_container:
            with right1:
                st.write("### Correlations")
                st.text(" Values closer to +1 indicate strong positive correlation.\n Values closer to -1 indicate strong negative correlation.\n Values closer to 0 indicate weak correlation.")
                corr_music = music_data.corr()
                y = list(corr_music.columns)
                x = list(corr_music.index)
                z = np.array(corr_music)

                fig = ff.create_annotated_heatmap(z, x = x, y = y, annotation_text = np.around(z, decimals=2),
                hoverinfo='z', colorscale='Teal')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('User didnot type any search term so no report can be compiled!')