
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import praw
import matplotlib.pyplot as plt
import math
import datetime as dt
import pandas as pd
import numpy as np
import os


nltk.download('vader_lexicon')
nltk.download('stopwords')

reddit = praw.Reddit(
    client_id= os.environ.get('reddit_id'),
    client_secret=os.environ.get('reddit_secret'),
    user_agent=os.environ.get('reddit_user_agent')
)



sub_reddits = reddit.subreddit('wallstreetbets')
stocks = ["SPCE", "LULU", "CCL", "SDC","TSLA"]


def commentSentiment(ticker, urlT):
    subComments = []
    bodyComment = []
    try:
        check = reddit.submission(url=urlT)
        subComments = check.comments
    except:
        return 0
    
    for comment in subComments:
        try: 
            bodyComment.append(comment.body)
        except:
            return 0
    
    sia = SIA()
    results = []
    for line in bodyComment:
        scores = sia.polarity_scores(line)
        scores['headline'] = line

        results.append(scores)
    
    df =pd.DataFrame.from_records(results)
    df.head()
    df['label'] = 0
    
    try:
        df.loc[df['compound'] > 0.1, 'label'] = 1
        df.loc[df['compound'] < -0.1, 'label'] = -1
    except:
        return 0
    
    averageScore = 0
    position = 0
    while position < len(df.label)-1:
        averageScore = averageScore + df.label[position]
        position += 1
    averageScore = averageScore/len(df.label) 
    
    return(averageScore)


def latestComment(ticker, urlT):
    subComments = []
    updateDates = []
    try:
        check = reddit.submission(url=urlT)
        subComments = check.comments
    except:
        return 0
    
    for comment in subComments:
        try: 
            updateDates.append(comment.created_utc)
        except:
            return 0
    
    updateDates.sort()
    return(updateDates[-1])



def get_date(date):
    return dt.datetime.fromtimestamp(date)



submission_statistics = []
d = {}
def research():
    for ticker in stocks:
        print(ticker)
        for submission in reddit.subreddit('wallstreetbets').search(ticker, limit=130):
            print(submission)
            if submission.domain != "self.wallstreetbets":
                continue
            d = {}
            d['ticker'] = ticker
            d['num_comments'] = submission.num_comments
            d['comment_sentiment_average'] = commentSentiment(ticker, submission.url)
            if d['comment_sentiment_average'] == 0.000000:
                continue
            d['latest_comment_date'] = latestComment(ticker, submission.url)
            d['score'] = submission.score
            d['upvote_ratio'] = submission.upvote_ratio
            d['date'] = submission.created_utc
            d['domain'] = submission.domain
            d['num_crossposts'] = submission.num_crossposts
            d['author'] = submission.author
            submission_statistics.append(d)
    

# research()


dfSentimentStocks = pd.DataFrame(submission_statistics)

_timestampcreated = dfSentimentStocks["date"].apply(get_date)
dfSentimentStocks = dfSentimentStocks.assign(timestamp = _timestampcreated)

_timestampcomment = dfSentimentStocks["latest_comment_date"].apply(get_date)
dfSentimentStocks = dfSentimentStocks.assign(commentdate = _timestampcomment)

dfSentimentStocks.sort_values("latest_comment_date", axis = 0, ascending = True,inplace = True, na_position ='last') 

dfSentimentStocks

dfSentimentStocks.author.value_counts()

dfSentimentStocks.to_csv('Reddit_Sentiment_Equity.csv', index=False) 



df = pd.read_csv('Reddit_Sentiment_Equity.csv')
df_tesla = df.loc[df['ticker'] == 'TSLA']

def toDatetime(df):
    """
    Casting "Object" colum to datetime
    """
#    df[['date','remove']] = df['timestamp'].str.split(' ', expand=True)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S') 

toDatetime(df_tesla)





from datetime import datetime

df_tesla = df_tesla.sort_values(by='timestamp_dt')


df_tesla.dtypes

# Selecting only rows after certain date
mask = df_tesla['timestamp_dt']>'2020-09-15 17:53:28'
dfDate = df_tesla.loc[mask]


import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(x=dfDate.timestamp_dt, y=dfDate.upvote_ratio))
fig.show()