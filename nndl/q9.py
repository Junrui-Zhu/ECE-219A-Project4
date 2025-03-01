import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
folder_path = "D:/ECE219_tweet_data"  # change it to your path
plot_data = {}
# for all .txt files in folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"): 
        file_path = os.path.join(folder_path, filename)
        print(f"dealing with: {filename}")
        
        timestamps = []
        follower_counts = []
        retweet_counts = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                tweet = json.loads(line.strip())  # load JSON
                if 'citation_date' in tweet:
                    timestamps.append(tweet['citation_date'])  # time stamp of tweets
                if 'author' in tweet and 'followers' in tweet['author']:
                    follower_counts.append(tweet['author']['followers'])  # number of followers
                if 'metrics' in tweet and 'citations' in tweet['metrics']:
                    retweet_counts.append(tweet['metrics']['citations']['total'])  # number of citations


        # convert time stamp to readable date
        if timestamps:
            df = pd.DataFrame({'timestamp': timestamps})
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['hour'] = df['datetime'].dt.floor('h') 
            
            tweet_counts = df.groupby('hour').size()
            avg_tweets_per_hour = tweet_counts.mean()
            print(f"Average number of tweets per hour: {avg_tweets_per_hour:.2f}")
            
            avg_followers_per_tweet = sum(follower_counts) / len(follower_counts) if follower_counts else 0
            print(f"Average number of followers of users posting the tweets per tweet: {avg_followers_per_tweet:.2f}")
            
            avg_retweets_per_tweet = sum(retweet_counts) / len(retweet_counts) if retweet_counts else 0
            print(f"Average number of retweets per tweet: {avg_retweets_per_tweet:.2f}\n")
            if filename == "tweets_#superbowl.txt" or filename=="tweets_#nfl.txt":
                plot_data[filename] = tweet_counts
        else:
            print(f"{filename} - no valid time stamp。")

if plot_data:
    plt.figure(figsize=(12, 6))
    for filename, tweet_counts in plot_data.items():
        plt.plot(tweet_counts.index, tweet_counts.values, label=filename.replace(".txt", ""))
    
    plt.xlabel("Time (Hourly)")
    plt.ylabel("Number of Tweets")
    plt.title("Number of Tweets per Hour for NFL and SuperBowl")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()