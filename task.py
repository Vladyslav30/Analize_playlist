from googleapiclient.discovery import build
import pandas as pd
from IPython.display import JSON
from dateutil import parser
import isodate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set(style="darkgrid", color_codes=True)
import seaborn as sns
# Google API
from googleapiclient.discovery import build
# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from wordcloud import WordCloud
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import numpy as np
# from matplotlib.backends.backend_pdf import PdfFileWriter, PdfFileReader


api_key = 'AIzaSyCDEEqn35XQVXRvAx9hSeLLThjTDBeB78A'
channel_ids = ["UCNED_pt9ZDfj6s_zT7FmAiA"]
api_service_name = "youtube"
api_version = "v3"
youtube = build(
    api_service_name, api_version, developerKey=api_key)


def get_channel_stats(youtube, channel_ids):
    all_data = []

    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=",".join(channel_ids)
    )
    response = request.execute()

    for item in response['items']:
        data = {'channelName': item['snippet']['title'],
                'subscribers': item['statistics']['subscriberCount'],
                'views': item['statistics']['viewCount'],
                'totalVideos': item['statistics']['videoCount'],
                'playlistId': item['contentDetails']['relatedPlaylists']['uploads']
                }
        all_data.append(data)

        return pd.DataFrame(all_data)

channel_stats = get_channel_stats(youtube, channel_ids)


def get_video_ids(youtube, playlist_id):
    request = youtube.playlistItems().list(
        part='contentDetails',
        playlistId=playlist_id,
        maxResults=50)
    response = request.execute()

    video_ids = []

    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])

    next_page_token = response.get('nextPageToken')
    more_pages = True

    while more_pages:
        if next_page_token is None:
            more_pages = False
        else:
            request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token)
            response = request.execute()

            for i in range(len(response['items'])):
                video_ids.append(response['items'][i]['contentDetails']['videoId'])

            next_page_token = response.get('nextPageToken')

    return video_ids

playlist_id = "PLTjRvDozrdlxj5wgH4qkvwSOdHLOCx10f"
video_ids = get_video_ids(youtube, playlist_id)


def get_video_details(youtube, video_ids):
    """
    Get video statistics of all videos with given IDs
    Params:

    youtube: the build object from googleapiclient.discovery
    video_ids: list of video IDs

    Returns:
    Dataframe with statistics of videos, i.e.:
        'channelTitle', 'title', 'description', 'tags', 'publishedAt'
        'viewCount', 'likeCount', 'favoriteCount', 'commentCount'
        'duration', 'definition', 'caption'
    """

    all_video_info = []

    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(video_ids[i:i + 50])
        )
        response = request.execute()

        for video in response['items']:
            stats_to_keep = {'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],
                             'statistics': ['viewCount', 'likeCount', 'favouriteCount', 'commentCount'],
                             'contentDetails': ['duration', 'definition', 'caption']
                             }
            video_info = {}
            video_info['video_id'] = video['id']

            for k in stats_to_keep.keys():
                for v in stats_to_keep[k]:
                    try:
                        video_info[v] = video[k][v]
                    except:
                        video_info[v] = None

            all_video_info.append(video_info)

    return pd.DataFrame(all_video_info)

video_df = get_video_details(youtube, video_ids)


def get_comments_in_videos(youtube, video_ids):
    """
    Get top level comments as text from all videos with given IDs (only the first 10 comments due to quote limit of Youtube API)
    Params:

    youtube: the build object from googleapiclient.discovery
    video_ids: list of video IDs

    Returns:
    Dataframe with video IDs and associated top level comment in text.

    """
    all_comments = []

    for video_id in video_ids:
        try:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id
            )
            response = request.execute()

            comments_in_video = [comment['snippet']['topLevelComment']['snippet']['textOriginal'] for comment in
                                 response['items']]
            comments_in_video_info = {'video_id': video_id, 'comments': comments_in_video}

            all_comments.append(comments_in_video_info)

        except:
            # When error occurs - most likely because comments are disabled on a video
            print('Could not get comments for video ' + video_id)

    return pd.DataFrame(all_comments)


comments_df = get_comments_in_videos(youtube, video_ids)


#Convert count columns to numeric columns
numeric_cols = ['viewCount', 'likeCount', 'commentCount']
# Ensure 'publishedAt' is in datetime format
video_df['publishedAt'] = pd.to_datetime(video_df['publishedAt'])

# Now you can extract the day name
video_df['durationSecs'] = video_df['duration'].apply(lambda x: isodate.parse_duration(x).total_seconds())
video_df['commentCount'] = pd.to_numeric(video_df['commentCount'])
video_df['likeCount'] = pd.to_numeric(video_df['likeCount'])

video_df['tagCount'] = video_df['tags'].apply(lambda x: 0 if x is None else len(x))
# Convert 'viewCount' column to numeric
video_df['viewCount'] = pd.to_numeric(video_df['viewCount'])
video_df['publishedAt'] = pd.to_datetime(video_df['publishedAt'])
video_df['publishedDay'] = video_df['publishedAt'].dt.day_name()

## how use sublot configuration tool setting
# Create a figure with two subplots



fig1, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
plt.subplots_adjust(bottom=0.557, top=0.964, left=0.125, right=0.9)
# Create the first bar plot
sns.barplot(x='title', y='viewCount', data=video_df.sort_values("viewCount", ascending=False)[0:9], ax=axes[0])
axes[0].set_title('Top 10 videos by view count')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K' if x>1000 and x<1000000 else '{:,.0f}'.format(x / 1000000) + 'M' if x>=1000000 else x))

# Create the second bar plot
sns.barplot(x = 'title', y = 'viewCount', data = video_df.sort_values('viewCount', ascending=True)[0:9], ax=axes[1])
axes[1].set_title('Bottom 10 videos by view count')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos:'{:,.0f}'.format(x/1000) + 'K' if x>1000 and x<1000000 else '{:,.0f}'.format(x/1000000) + 'M' if x>=1000000 else x))






def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
fig2, ax = plt.subplots(1,2, figsize=(20,10))

# First scatter plot
sns.scatterplot(data=video_df, x='commentCount', y='viewCount', ax=ax[0])
ax[0].set_title('View count vs Comment count')
ax[0].set_xlabel('Comment Count')
ax[0].set_ylabel('View Count')
ax[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: human_format(x)))
ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: human_format(x)))
ax[0].set_xticklabels(ax[0].get_xticks())

# Second scatter plot
sns.scatterplot(data=video_df, x='likeCount', y='viewCount', ax=ax[1])
ax[1].set_title('View count vs Like count')
ax[1].set_xlabel('Like Count')
ax[1].set_ylabel('View Count')
ax[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: human_format(x)))
ax[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: human_format(x)))
ax[1].set_xticklabels(ax[1].get_xticks())
#
# fig4, ax = plt.subplots(figsize=(10,5))
# sns.violinplot(x=video_df['channelTitle'], y=video_df['viewCount'], ax=ax)
# ax.set_title('Distribution of View Count by Channel')
# ax.set_xlabel('Channel')


fig5, ax = plt.subplots(figsize=(20,10))
sns.histplot(data = video_df, x = 'durationSecs', bins=30)
ax.set_title('Distribution of Video Duration')
ax.set_xlabel('Duration (seconds)')



weekday_counts = video_df['publishedDay'].value_counts().to_dict()
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_df = pd.DataFrame(list(weekday_counts.items()), columns=['Day', 'Count'])
day_df['Day'] = pd.Categorical(day_df['Day'], categories=weekdays, ordered=True)
day_df = day_df.sort_values('Day')
fig6, ax = plt.subplots(figsize=(20, 10))
sns.barplot(x='Day', y='Count', data=day_df, ax=ax, palette='viridis')
ax.set_title('Number of Videos Published on Each Day of the Week')
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Number of Videos')

# Download the required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Set up Russian stopwords
stop_words = set(stopwords.words('english'))

# Apply stopwords removal to title column
video_df['title_no_stopwords'] = video_df['title'].apply(lambda x: [item for item in word_tokenize(str(x)) if item.lower() not in stop_words])
comments_df['comments_no_stopwords'] = comments_df['comments'].apply(lambda x: [item for item in str(x).split() if item not in stop_words])
all_words_2 = list([a for b in comments_df['comments_no_stopwords'].tolist() for a in b])
all_words_str_2 = ' '.join(all_words_2)

# Flatten the list of words
all_words = [word for sublist in video_df['title_no_stopwords'] for word in sublist]
all_words_str = ' '.join(all_words)

def plot_cloud(wordcloud):
    plt.figure(figsize=(30, 20))
    plt.imshow(wordcloud)
    plt.axis("off")

# Generate and display the word cloud
wordcloud = WordCloud(width=2000, height=1000, random_state=1, background_color='black',
                      colormap='viridis', collocations=False).generate(all_words_str)
plot_cloud(wordcloud)
wordcloud_2 = WordCloud(width = 2000, height = 1000, random_state=1, background_color='black',
                      colormap='viridis', collocations=False).generate(all_words_str_2)
plot_cloud(wordcloud_2)




wordcloud.to_file('wordcloud.png')
img = mpimg.imread('wordcloud.png')
fig3, ax = plt.subplots(figsize=(20,10))
ax.imshow(img)
ax.axis('off')
wordcloud_2.to_file('wordcloud_2.png')
img = mpimg.imread('wordcloud_2.png')
fig7, ax = plt.subplots(figsize=(20,10))
ax.imshow(img)
ax.axis('off')


with PdfPages('output.pdf') as pdf:
    fig1.savefig(pdf, format='pdf')  # Save figure 1
    fig2.savefig(pdf, format='pdf')  # Save figure 2
    fig3.suptitle('key frases in title of videos') # Add a title so we know which it is
    fig3.savefig(pdf, format='pdf')  # Save figure 3
    # fig4.savefig(pdf, format='pdf')  # Save figure 4
    fig5.savefig(pdf, format='pdf')  # Save figure 5
    fig6.savefig(pdf, format='pdf')  # Save figure 6
    fig7.suptitle('key frases in comment below videos') # Add a title so we know which it is
    fig7.savefig(pdf, format='pdf')  # Save figure 7


