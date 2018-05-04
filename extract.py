import got.manager
import csv

print("start setting")
tweetCri = got.manager.TweetCriteria().setQuerySearch('trump').setSince('2018-04-02').setUntil('2018-05-02').setMaxTweets(10)
print('start searching')
tweets = got.manager.TweetManager.getTweets(tweetCri)

with open('tweets_test.csv', 'w') as tweets_file:
    writer = csv.writer(tweets_file, delimiter=',')
    for tweet in tweets:
        # attrs = vars(tweet)

        # print('\n'.join("%s : %s " % item for item in attrs.items()))
        tweet_txt = tweet.text
        tweet_txt = tweet_txt.replace(u"\u2026", "...")  # ...
        tweet_txt = tweet_txt.replace(u"\u201c", "\"")  # "
        tweet_txt = tweet_txt.replace(u"\u201d", "\"")  # "
        tweet_txt = tweet_txt.replace(u"\u2018", "'")  # '
        tweet_txt = tweet_txt.replace(u"\u2019", "'")  # '
        tweet_txt = tweet_txt.replace(u"\u2014", "-")  # dash
        writer.writerow([tweet.id, tweet.permalink, tweet.username, tweet_txt, tweet.date, tweet.retweets, tweet.favorites, tweet.mentions, tweet.hashtags, tweet.geo])
