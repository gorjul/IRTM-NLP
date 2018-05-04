# coding=utf-8
from nltk.corpus import stopwords
import re
import pandas as pd
import csv
from textblob import TextBlob


def clean_tweet(tweet):
    """
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def analize_sentiment(tweet):
    """
    Utility function to classify the polarity of a tweet
    using textblob.
    """
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


frequency = {}

print("read tweets")
df = pd.read_csv('tweets_test.csv', names=['id', 'permalink', 'username', 'tweet', 'date', 'retweets', 'favorites', 'mentions', 'hashtags', 'geo'],
                 na_values=['.']).as_matrix()

stop = set(stopwords.words('english'))
stop.add(".")

print("process tweets")
with open('process.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for x in range(0, len(df)):
        # tweet.id(0), tweet.permalink(1), tweet.username(2), tweet_txt(3), tweet.date(4), tweet.retweets(5), tweet.favorites(6), tweet.mentions(7), tweet.hashtags(8), tweet.geo(9)
        tweet = str(df[x][3])

        # print(type(tweet))

        pos = max(tweet.find("http"), tweet.find("https"))
        url = ""
        if pos > -1:
            count_whitespace = 0
            for i in range(pos, len(tweet)):
                if tweet[i] in " ":
                    count_whitespace = count_whitespace + 1
                if count_whitespace is 3:
                    break
                url += tweet[i]
        tweet = tweet.replace(url, "")  # re.sub(r"\s", "", url))
        tweet = re.sub(r"pic\.twitter\.com\/\w*", "", tweet)
        tweet = re.sub(r"n't", " not", tweet)
        tweet = re.sub(r"'ll", " will", tweet)
        tweet = re.sub(r"[!\?,\"><()]", "", tweet)
        tweet = re.sub(r"\s-\s", " ", tweet)
        tweet = re.sub(r"\s\.\s|\.\s|\s\.|\.{3}", " ", tweet)
        tweet = re.sub(r"\.$", "", tweet)
        # print(tweet)
        writer.writerow([df[x][0], df[x][1], df[x][2], tweet, df[x][4], df[x][5], df[x][6], df[x][7], df[x][8], df[x][9]])

        for word in tweet.split():
            count = frequency.get(word, 0)
            frequency[word] = count + 1

print("create frequency")
frequency = {k: v for k, v in frequency.items() if v > 1 and k not in stop}

with open('frequency.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in frequency.items():
        writer.writerow([key, value])
