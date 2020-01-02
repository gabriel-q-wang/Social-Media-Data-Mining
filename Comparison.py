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

        # attempt authentication
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
            return follower_list

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

    @staticmethod
    def star_rating(followers):
        # Rating system for followers
        # 1 million+ is 4 stars, considered a top influencer
        # 500000+ is 3 stars, considered an influencer
        # 100000+ is 2 stars, considered a rising influencer
        # All others are considered normal users
        # Note: This can be changed depending on your standards
        # Using the current standards, more than 99% of users are considered normal users
        if followers >= 1000000:
            return 4
        elif followers >= 500000:
            return 3
        elif followers >= 100000:
            return 2
        else:
            return 1

    def rate_followers(self, follower_list):
        # Record info about a user's followers and rate them depending on their followers
        df = pd.DataFrame(columns=['user', 'screen_name', 'verified', 'followers', 'following', 'star'])
        try:
            follower_list = [str(i) for i in follower_list]
            df['user'] = follower_list
            df = df.set_index('user')
            for users in follower_list:
                try:
                    user = self.api.get_user(users)
                    df.at[str(user.id), 'screen_name'] = user.screen_name
                    df.at[str(user.id), 'verified'] = user.verified
                    df.at[str(user.id), 'followers'] = user.followers_count
                    df.at[str(user.id), 'following'] = user.friends_count
                    df.at[str(user.id), 'star'] = self.star_rating(user.followers_count)
                except:
                    print('User is banned or cannot be found')
            df = df.reset_index()
            return df

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))
            return df


def main():
    # creating object of TwitterClient Class
    api = TwitterClient()

    # Get a list of followers
    Haworthinc_followers_list = api.get_following('Haworthinc')
    # Create a dataframe of resulting data
    Haworthinc_followers_df = api.rate_followers(Haworthinc_followers_list)
    Haworthinc_followers_df.sort_values(by=['star'], ascending=False)
    print(Haworthinc_followers_df.head())
    # Export to csv
    # Note: Change file path
    Haworthinc_followers_df.to_csv("C:\\Users\\gwang\\Documents\\01 ADS Projects\\Haworthinc_influencers_test.csv", index=False)


if __name__ == "__main__":
    # calling main function
    main()
