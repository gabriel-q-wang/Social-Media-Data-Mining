import config
import re
import tweepy
import pandas as pd
import requests
import base64
import json
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class TwitterSentiment(object):
    # Generic Twitter Class for sentiment analysis.

    def __init__(self):
        # Class constructor or initialization method.
        # keys and tokens from the Twitter Dev Console
        # stored in a config file
        consumer_key = config.consumer_key
        consumer_secret = config.consumer_secret

        # Encode keys and secrets for proper login
        key_secret = '{}:{}'.format(consumer_key, consumer_secret).encode('ascii')
        b64_encoded_key = base64.b64encode(key_secret)
        b64_encoded_key = b64_encoded_key.decode('ascii')
        base_url = 'https://api.twitter.com/'
        auth_url = '{}oauth2/token'.format(base_url)

        auth_headers = {
            'Authorization': 'Basic {}'.format(b64_encoded_key),
            'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
        }

        auth_data = {
            'grant_type': 'client_credentials'
        }

        # Get access token and login
        auth_resp = requests.post(auth_url, headers=auth_headers, data=auth_data)
        print(auth_resp.status_code)
        auth_resp.json().keys()
        access_token = auth_resp.json()['access_token']
        self.search_headers = {
            'Authorization': 'Bearer {}'.format(access_token)
        }

        # Download stopwords and set up vectorizer for later text analysis
        # Note: This code currently only works for English
        nltk.download('stopwords')
        self.trained_model = None
        self.stop = stopwords.words('english')
        self.tfidf = TfidfVectorizer(strip_accents=None,
                                     lowercase=False,
                                     preprocessor=None,
                                     tokenizer=self.tokenizer_porter,
                                     stop_words=self.stop)

    @staticmethod
    def clean_tweet(tweet):
        # Utility function to clean tweet text by removing links, special characters using simple regex statements.
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split())

    @staticmethod
    def tokenizer_porter(text):
        # Creates a porter for later text analysis
        porter = PorterStemmer()
        return [porter.stem(word) for word in text.split()]

    def train_logistic_regression(self):
        # Trains a logistic regression model to predict text sentiment
        # Note: Logistic regression isn't that accurate, this model only has an accuracy in the high 70s
        # A deep neural network is recommended, use tools from Azure, AWS, etc.

        # Read in training data
        # https://www.kaggle.com/kazanova/sentiment140
        # Note: The data only has positive(4) and negative(0) sentiment, no neutrals
        df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1',
                         names=["target", "ids", "date", "flag", "user", "text"])
        df = df.filter(items=['target', 'text'])
        df.columns = ['sentiment', 'review']

        # Set sentiment so that it's 1 for positive, 0 for negative
        df.sentiment = df.sentiment.replace(4, 1)
        df.review = df.review.apply(self.clean_tweet)
        np.random.seed(0)
        df = df.reindex(np.random.permutation(df.index))
        print(df.head(10))

        # Split data into testing and training datasets
        word_frequency = self.tfidf.fit_transform(df.review)
        sample_index = np.random.random(df.shape[0])
        X_train, X_test = word_frequency[sample_index <= 0.8, :], word_frequency[sample_index > 0.8, :]
        Y_train, Y_test = df.sentiment[sample_index <= 0.8], df.sentiment[sample_index > 0.8]
        print(f"shape of training set: X={X_train.shape}, Y={Y_train.shape}")
        print(f"shape of test set: X={X_test.shape}, Y={Y_test.shape}")

        # Create, fit, and test logistic regression model
        clf = LogisticRegression(random_state=0, solver='saga', multi_class='multinomial').fit(X_train, Y_train)
        Y_predict = clf.predict(X_test)
        print("Accuracy: " + str(sum(Y_predict == Y_test) / len(Y_test)))
        print('Confusion Matrix:')
        print(confusion_matrix(Y_test,Y_predict))
        print('Classification Report:')
        print(classification_report(Y_test, Y_predict))

        # Set the model
        self.trained_model = clf

    def get_tweet_sentiment_logistic_regression(self, tweet):
        # Gets the predicted sentiment from a cleaned tweet
        tweet_convert = self.tfidf.transform([tweet])
        pred = self.trained_model.predict(tweet_convert)
        return pred

    @staticmethod
    def add_to_dataframe(tweet, df, parsed_tweet):
        # Add the data from the API to the dataframe
        user_id = str(tweet.get('user').get('id'))
        df = df.append({'user_id': user_id}, ignore_index=True)
        df = df.set_index('user_id')
        df.at[user_id, 'user_name'] = tweet.get('user').get('name')
        df.at[user_id, 'date'] = tweet.get('created_at')
        df.at[user_id, 'sentiment'] = parsed_tweet.get('sentiment')
        df.at[user_id, 'tweet'] = parsed_tweet.get('text')
        df.at[user_id, 'link'] = 'https://twitter.com/i/web/status/' + tweet.get('id_str')
        df.at[user_id, 'location'] = tweet.get('user').get('location')
        df.at[user_id, 'follower_count'] = tweet.get('user').get('followers_count')
        df.at[user_id, 'friend_count'] = tweet.get('user').get('friends_count')
        df.at[user_id, 'verified'] = tweet.get('user').get('verified')
        df.at[user_id, 'retweet_count'] = tweet.get('retweet_count')
        df.at[user_id, 'favorite_count'] = tweet.get('favorite_count')
        df.at[user_id, 'reply_count'] = tweet.get('reply_count')
        df = df.reset_index()
        return df

    def get_dataframe(self, query):
        # Main function to fetch tweets and parse them.

        # Initialize dataframe
        tweets = []
        df = pd.DataFrame(columns=['user_id', 'user_name', 'tweet', 'link', 'sentiment', 'date', 'location',
                                   'follower_count', 'friend_count', 'retweet_count', 'favorite_count', 'reply_count',
                                   'verified'])
        try:
            # Next cursor is for getting the next page of results from the api
            next_cursor = -1

            while next_cursor is not None:
                if next_cursor == -1:
                    # Initial search
                    # Note: Change 30day to full archive if you want to search the full archive
                    # Also, change dev719 to whichever dev environment you have set up on your account
                    url = 'https://api.twitter.com/1.1/tweets/search/30day/dev719.json?query=%s' % query
                else:
                    # Search the next page
                    # Change dev719 to whichever dev environment you have set up on your account
                    url = 'https://api.twitter.com/1.1/tweets/search/30day/dev719.json?query=%s&next=%s' % \
                          (query, next_cursor)
                # Read and load results
                content = requests.get(url, headers=self.search_headers).content
                data = json.loads(content).get('results')
                next_cursor = json.loads(content).get('next')
                next_cursor = next_cursor[:-1]

                # parsing tweets one by one
                for tweet in data:
                    if 'text' in tweet:
                        # empty dictionary to store required params of a tweet
                        parsed_tweet = {}
                        full_text = tweet.get('text')
                        # saving text of tweet
                        if 'extended_tweet' in tweet:
                            full_text = tweet.get('extended_tweet').get('full_text')

                        # Parse and remove commas
                        full_text = full_text.replace(',', '')
                        parsed_tweet['text'] = str(self.clean_tweet(full_text))
                        # saving sentiment of tweet
                        parsed_tweet['sentiment'] = int(self.get_tweet_sentiment_logistic_regression(
                            self.clean_tweet(full_text)))

                        # appending parsed tweet to tweets list
                        if tweet.get('retweet_count') > 0:
                            # if tweet has retweets, ensure that it is appended only once
                            if parsed_tweet not in tweets:
                                df = self.add_to_dataframe(tweet, df, parsed_tweet)
                                tweets.append(parsed_tweet)
                        else:
                            df = self.add_to_dataframe(tweet, df, parsed_tweet)
                            tweets.append(parsed_tweet)

                        # return parsed tweets
                    elif 'message' in tweet:
                        print('%s (%d)' % (tweet['message'], tweet['code']))

            return df

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))
            return df

        except:
            # If an unknown error occurs, return the data you already have
            print('Error -------')
            return df


def main():
    # creating object of TwitterClient Class
    api = TwitterSentiment()

    # Train the logistic regression model
    api.train_logistic_regression()

    # Get a dataframe of resulting data from the search
    # Change query to whichever user you are interested in
    tweets = api.get_dataframe(query='Haworthinc')
    print(tweets.head())

    # Export as csv
    # Note: Change your path and names
    tweets.to_csv("C:\\Users\\gwang\\Documents\\01 ADS Projects\\sentiment_haworthinc_test.csv", index=False)


if __name__ == "__main__":
    # calling main function
    main()
