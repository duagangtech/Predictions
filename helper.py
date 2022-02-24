# Import required libraries
import datetime 
#import isodate
import pandas as pd


# Convert ISO time to duration in minutes
def ISO_to_duration(time):
    if len(time) == 5:
        initial = datetime.datetime.strptime(time, 'PT%SS')
        initial_time = str(initial.time())
        components= initial_time.split(':')
        total_duration = int(components[0]) * 60 + int(components[1]) +int(components[2])/60
    else:    
        try:
            initial = datetime.datetime.strptime(time, 'PT%HH%MM%SS')
        except:
            initial = datetime.datetime.strptime(time, 'PT%MM%SS')
        initial_time = str(initial.time())
        components= initial_time.split(':')
        total_duration = int(components[0]) * 60 + int(components[1]) +int(components[2])/60
    return (round(total_duration,2))

# Extracts the video ids from a json file into a list
def id_list(data_json):
    vid_ids = []
    for i, j in enumerate(data_json['items']):
        k = j['id']['videoId']
        vid_ids.append(k)
    #vid_id = pd.DataFrame({'ID':vid_ids})
    return(vid_ids)


# Takes in a JSON and extracts the features into a pandas dataframe

def video_data_extractor (data_you):
    # Initializing the lists
    release_date = []
    title = []
    channel_name = []
    total_views = []
    likes = []
    comments = []
    duration = []
    # loop through to save the features
    for i in data_you['items']:
        snippet = i['snippet']
        stats = i['statistics']
        content = i['contentDetails']
        # Separating the date
        try:
            # Q&A seem to have comment count missing
            comments.append(stats['commentCount'])
            likes.append(stats['likeCount'])
            # rest should be executed only if comment count also present 
            date_time = datetime.datetime.strptime(snippet['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
            date_time = date_time.strftime("%d-%m-%Y")
            release_date.append(date_time)
            title.append(snippet['title'])
            channel_name.append(snippet['channelTitle'])
            total_views.append(stats['viewCount'])
            #duration_delta = isodate.parse_duration(content['duration'])
            #duration_delta = round(duration_delta.seconds/60, 2)
            duration_minutes = ISO_to_duration(content['duration'])
            duration.append(duration_minutes)
        except:
            pass
    # Save as dataframe
    DF = pd.DataFrame({'title':title, 'Channel Name':channel_name, 'Release Date (Day-Month-Year)':release_date,'Duration (in minutes)':duration,\
        'Total Views':total_views, 'Total Likes':likes, 'Total Comments':comments})
    return(DF)
