import config
import tweepy
from tweepy import OAuthHandler
import pandas as pd


class TwitterClient(object):
    # Generic Twitter Class for sentiment analysis.

    def __init__(self):
        # Class constructor or initialization method.
        # keys and tokens from the Twitter Dev Console
        # stored in a config file
        consumer_key = config.consumer_key
        consumer_secret = config.consumer_secret
        access_token = config.access_token
        access_token_secret = config.access_token_secret

        # attempt authentication using Tweepy
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
        except:
            print("Error: Authentication Failed")

    def get_user(self, query):
        # Get user and information such as followers and tweets
        try:
            user = self.api.get_user(query)
            print("Screen name: " + str(user.screen_name))
            print("Verified: " + str(user.verified))
            print("Followers: " + str(user.followers_count))
            print("Following: " + str(user.friends_count))
            print("Lists: " + str(user.listed_count))
            print("Tweets + Retweets: " + str(user.statuses_count) + "\n")
            return user

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))

    def get_following(self, query):
        # Returns a list of all of a user's followers
        # Note: You will be rate limited during this action, there is no way around it
        # Be prepared to have a stable internet connect for a few hours or days depending on the size of the following
        try:
            follower_list = []
            for page in tweepy.Cursor(self.api.followers_ids, screen_name=query).pages():
                follower_list.extend(page)
            return follower_list

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))

    def get_following_number(self, query):
        # Return an integer of the number of followers
        try:
            user = self.api.get_user(query)
            return user.followers_count

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))

    def get_friends(self, query):
        # Returns a list of all of a user's friends
        # Note: You will be rate limited during this action, there is no way around it
        # Be prepared to have a stable internet connect for a few hours or days depending on the size of the following
        try:
            friend_list = []
            for page in tweepy.Cursor(self.api.friends_ids, screen_name=query).pages():
                friend_list.extend(page)
            return friend_list

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))

    def get_friends_number(self, query):
        # Return an integer of the number of friends
        try:
            user = self.api.get_user(query)
            return user.friends_count

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))

    def compare_users_followers(self, query1, query2):
        # Return a percent of mutual followers between two users
        # Note: You will be rate limited during this action, there is no way around it
        # Be prepared to have a stable internet connect for a few hours or days depending on the size of the following
        try:
            user1 = self.get_following(query1)
            user2 = self.get_following(query2)
            intersection = [value for value in user1 if value in user2]
            return len(intersection)

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))

    def compare_users_friends(self, query1, query2):
        # Return a percent of mutual followers between two users
        # Note: You will be rate limited during this action, there is no way around it
        # Be prepared to have a stable internet connect for a few hours or days depending on the size of the following
        try:
            user1 = self.get_friends(query1)
            user2 = self.get_friends(query2)
            intersection = [value for value in user1 if value in user2]
            return len(intersection)

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))

    def set_user_info(self, query, df):
        # Returns a dataframe of a user's followers, tweets, etc
        user = self.get_user(query=query)
        df.at[query, 'verified'] = user.verified
        df.at[query, 'followers'] = user.followers_count
        df.at[query, 'following'] = user.friends_count
        df.at[query, 'lists'] = user.listed_count
        df.at[query, 'tweets'] = user.statuses_count
        return df

    def set_compare_info(self, query1, query2, df):
        # Set information in a dataframe about mutual followers and friends
        # Note: You will be rate limited during this action, there is no way around it
        # Be prepared to have a stable internet connect for a few hours or days depending on the size of the following

        mutual_followers = self.compare_users_followers(query1, query2)
        mutual_friends = self.compare_users_friends(query1, query2)
        # Number of mutual followers
        df.at[query1, query1 + '_number_followers'] = -1
        df.at[query1, query2+'_number_followers'] = mutual_followers
        df.at[query2, query2 + '_number_followers'] = -1
        df.at[query2, query1 + '_number_followers'] = mutual_followers
        # Percent mutual followers
        df.at[query1, query1 + '_percent_followers'] = -1
        df.at[query1, query2 + '_percent_followers'] = (mutual_followers / self.get_following_number(query1))
        df.at[query2, query2 + '_percent_followers'] = -1
        df.at[query2, query1 + '_percent_followers'] = (mutual_followers / self.get_following_number(query2))
        # Number of mutual followers
        df.at[query1, query1 + '_number_friends'] = -1
        df.at[query1, query2 + '_number_friends'] = mutual_friends
        df.at[query2, query2 + '_number_friends'] = -1
        df.at[query2, query1 + '_number_friends'] = mutual_friends
        # Percent of mutual friends
        df.at[query1, query1 + '_percent_friends'] = -1
        df.at[query1, query2 + '_percent_friends'] = (mutual_friends / self.get_friends_number(query1))
        df.at[query2, query2 + '_percent_friends'] = -1
        df.at[query2, query1 + '_percent_friends'] = (mutual_friends / self.get_friends_number(query2))

        return df


def main():
    # creating object of TwitterClient Class
    api = TwitterClient()
    # Initialize dataframe
    df = pd.DataFrame(columns=['user', 'verified', 'followers', 'following', 'lists', 'tweets',
                               'Steelcase_number_followers', 'HermanMiller_number_followers',
                               'Haworthinc_number_followers', 'Steelcase_percent_followers',
                               'HermanMiller_percent_followers', 'Haworthinc_percent_followers',
                               'Steelcase_number_friends', 'HermanMiller_number_friends',
                               'Haworthinc_number_friends', 'Steelcase_percent_friends',
                               'HermanMiller_percent_friends', 'Haworthinc_percent_friends'])

    user_list = ['Steelcase', 'HermanMiller', 'Haworthinc']
    df['user'] = user_list
    df = df.set_index('user')
    # Set info about number of followers and tweets
    df = api.set_user_info("Steelcase", df)
    df = api.set_user_info("HermanMiller", df)
    df = api.set_user_info("Haworthinc", df)
    # Set compared info between users
    df = api.set_compare_info("Steelcase", "HermanMiller", df)
    df = api.set_compare_info('Steelcase', "Haworthinc", df)
    df = api.set_compare_info("HermanMiller", 'Haworthinc', df)

    df = df.reset_index()
    print(df.head())

    # Export to csv
    # Note: Change your file path
    df.to_csv("C:\\Users\\gwang\\Documents\\01 ADS Projects\\Mutual_Comparison.csv", index=False)


if __name__ == "__main__":
    # calling main function
    main()
