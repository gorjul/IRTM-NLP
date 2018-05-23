# coding: utf-8
import got.manager
import csv
from timeit import default_timer as timer
import datetime
from datetime import datetime
from datetime import timedelta

def search_and_write(file_name, day_before, monday):

    start_time = datetime.fromtimestamp(timer())
    print("Start collecting Tweets at %s" % start_time.strftime('%H:%M:%S'))
    tweetCri = got.manager.TweetCriteria().setUsername('realDonaldTrump').setSince(str(day_before.date())).setUntil(str(monday.date())).setMaxTweets(100)
    # tweetCri = got.manager.TweetCriteria().setQuerySearch('trump').setSince(str(day_before.date())).setUntil(str(monday.date())).setMaxTweets(10000)
    print('start searching', monday.date())
    tweets = got.manager.TweetManager.getTweets(tweetCri)

    end_time_tweets = datetime.fromtimestamp(timer())

    print("End collecting Tweets at %s" % end_time_tweets.strftime('%H:%M:%S'))

    with open(file_name, 'w') as tweets_file:
        writer = csv.writer(tweets_file, delimiter=',')

        for tweet in tweets:
            writer.writerow([tweet.id, tweet.permalink, tweet.username, tweet.text.encode('utf8'), tweet.date, tweet.retweets, tweet.favorites, tweet.mentions, tweet.hashtags, tweet.geo])

    end_time = datetime.fromtimestamp(timer())

    print("End writing Tweets at %s" % end_time.strftime('%H:%M:%S'))


monday = datetime(2017, 10, 30, 00, 00)  # first monday
day_before_monday = datetime(2017, 10, 29, 00, 00)  # first monday

for count in range(26):
    print(monday)
    search_and_write('users/tweets_test_monday_' + str(monday.date()) + '.csv', day_before_monday, monday)
    # search_and_write('trump/tweets_test_monday_trump_' + str(monday.date()) + '.csv', day_before_monday, monday)
    monday = monday + timedelta(days=7)
    day_before_monday = day_before_monday + timedelta(days=7)
