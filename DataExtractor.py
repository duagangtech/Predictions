import json
from sys import api_version, argv
#import datetime as dt
from helper import *

# API client library
from tkinter import EventType
import googleapiclient.discovery

# Import API Keys
with open('youtube_credentials.json') as f:    
    cred = json.load(f)

api_key = cred['api_key']
api_version = cred["api_version"]
api_service_name = cred["api_service_name"]

# API client
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey = api_key)


def query_ids(search_term):
    """
    query_ids (search_term) takes in a search_term and returns the ids of the top 50 most relevant results.
    query_ids : String -> List
    Note: Might return less than 50 results if data not available
    """
    request = youtube.search().list(
        part ="id",
        order ='relevance', 
        q = search_term,
        maxResults = 50, # Max 50
        safeSearch = "none",  # "strict",
        type = "video",
        eventType = "completed",
        fields = "items(id(videoId))"
        #publishedAfter =  The value is an RFC 3339 formatted date-time value (1970-01-01T00:00:00Z).
        #publishedBefor = The value is an RFC 3339 formatted date-time value (1970-01-01T00:00:00Z).
    ).execute()
    id_results = id_list(request)
    return(id_results)

def video_data(ids_from_query):
    """
    video_data(ids_from_query) takes in a kist of ids retrieved from the YouTube API and extracts all the releavnt features
    video_data: List -> Pandas Dataframe
    """
    request = youtube.videos().list(
        part ="contentDetails, statistics, snippet",
        regionCode = "CA",
        id = ids_from_query,
        fields = "items(snippet(title, publishedAt, channelId, channelTitle)" + "statistics," + "contentDetails(duration))"
        ).execute()
    # Converts JSON to DataFrame
    data_vid = video_data_extractor(request)
    return (data_vid)


def wrapper_data(search_term):
    """
    wrapper_data (search_term) takes in search_term ro retrive the features of the top 50 most relevant results
    and saves them in a csv file
    wrapper_data: String -> None
    """
    ids_list = query_ids(search_term)
    data_from_API = video_data(ids_list)
    data_from_API.to_csv('music.csv',index=False)

if __name__ == '__main__':
    try:
        wrapper_data(argv[1])
    except:
        pass