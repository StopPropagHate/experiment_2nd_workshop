from twython import Twython
from twython import TwythonError,TwythonRateLimitError
#https://github.com/ryanmcgrath/twython
import os
import json
import re
from time import sleep

PATH_TO_PF_DATASET = "C:\/Users\/Ze\/Desktop\/pf_hate_speech\/"
PATH_TO_DATASET_FILE = "datasetdummyclasses.csv"

# GET Credentials
def get_credentials():
    with open("twitter_credentials.json", "r") as file:
        creds = json.load(file)
    file.close()
    return creds

# GET tweets_data
def get_tweets_data():
    with open("tweets_data.json", "r") as file:
        tweets_data = json.load(file)
    file.close()
    return tweets_data

# GET Tweet Ids
def get_tweets_ids():
    tweets = []
    path = PATH_TO_PF_DATASET + PATH_TO_DATASET_FILE
    if (os.path.exists(path)):
        with open(path, "r") as file:
            lines = file.readlines()
            header = lines.pop()
            for line in lines:
                id = re.findall('^(\d+),.*\n',line)
                if id != []:
                    tweets.append(id.pop())
        file.close()
    return tweets

# GET tweets Missing
def get_missing_tweets(tweets_ids,tweets_data):
    missing_tweets= []
    for t in tweets_ids:
        if t not in tweets_data.keys():
            missing_tweets.append(t)


#tweets_ids = get_tweets_ids()
tweets_data = get_tweets_data()
#missing_tweets = get_missing_tweets(tweets_ids,tweets_data)

# fetch tweets
def fetch_tweets(twitter,missing_tweets):
    tweets_not_found = []
    tweets_forbidden = []
    tweets_too_many_requests = []

    for tweet in missing_tweets:
        try:
            status = twitter.show_status(id=tweet)
            tweets_data[tweet] = status
        except TwythonError as error:
            # Not Found
            if error.error_code == 404:
                tweets_not_found.append(tweet)
                print(error.msg, error.error_code)
            #Forbidden
            elif error.error_code == 403:
                tweets_forbidden.append(tweet)
                print(error.msg, error.error_code)
            #Too Many Requests
            elif error.error_code == 429:
                sleep_time = 10
                sleep(sleep_time)
                tweets_too_many_requests.append(tweet)
                print(error.msg, error.error_code)

    print('tweets_not_found:', len(tweets_not_found))
    print('tweets_forbidden:', len(tweets_forbidden))
    print('tweets_too_many_requests:',len(tweets_too_many_requests))

    return tweets_data


#creds = get_credentials()
#twitter = Twython(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
#missing_data = fetch_tweets(twitter,missing_tweets)
#tweets_data.update(missing_data)

#save_tweets_data(tweets_data)

def save_tweets_data(tweets_data):
    with open("tweets_data.json", "w") as file:
        json.dump(tweets_data, file)
    file.close()

# 3135

