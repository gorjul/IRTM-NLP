# coding: utf-8
import got.manager
import csv
from timeit import default_timer as timer
from datetime import datetime

dates_monday = {
    # '2018-04-01': '2018-04-02',
    # '2018-04-08': '2018-04-09',
    '2018-04-15': '2018-04-16',
    # '2018-04-22': '2018-04-23',
    '2018-04-29': '2018-04-30'
}
# dates_wednesday = {'2018-04-03': '2018-04-04', '2018-04-10': '2018-04-11', '2018-04-17': '2018-04-18', '2018-04-24': '2018-04-25', '2018-05-01': '2018-05-02'}
# dates_friday = {'2018-04-05': '2018-04-06', '2018-04-12': '2018-04-13', '2018-04-19': '2018-04-20', '2018-04-26': '2018-04-27', '2018-05-03': '2018-05-04'}


def search_and_write(file_name, s, u):

    start_time = datetime.fromtimestamp(timer())
    print("Start collecting Tweets at %s" % start_time.strftime('%H:%M:%S'))
    tweetCri = got.manager.TweetCriteria().setQuerySearch('trump').setSince(s).setUntil(u).setMaxTweets(10000)
    print('start searching', until)
    tweets = got.manager.TweetManager.getTweets(tweetCri)

    end_time_tweets = datetime.fromtimestamp(timer())

    print("End collecting Tweets at %s" % end_time_tweets.strftime('%H:%M:%S'))

    with open(file_name, 'w') as tweets_file:
        writer = csv.writer(tweets_file, delimiter=',')

        for tweet in tweets:
            writer.writerow([tweet.id, tweet.permalink, tweet.username, tweet.text.encode('utf8'), tweet.date, tweet.retweets, tweet.favorites, tweet.mentions, tweet.hashtags, tweet.geo])

    end_time = datetime.fromtimestamp(timer())

    print("End writing Tweets at %s" % end_time.strftime('%H:%M:%S'))


print('Mondays')
for since in dates_monday:
    until = dates_monday[since]
    search_and_write('tweets_test_monday_' + until + '.csv', since, until)


# print('Wednesdays')
# for since in dates_wednesday:
#     until = dates_wednesday[since]
#     search_and_write('tweets_test_wednesday_' + until + '.csv', since, until)
#
#
# print('Fridays')
# for since in dates_friday:
#     until = dates_wednesday[since]
#     search_and_write('tweets_test_friday_' + until + '.csv', since, until)
