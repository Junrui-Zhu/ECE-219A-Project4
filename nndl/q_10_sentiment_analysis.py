import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import ast
import numpy as np

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish')) | set(stopwords.words('english'))

def read_brand(brand_path='dataset/brand.txt'):
    with open(brand_path, "r", encoding="utf-8") as file:
        brand_list = [line.strip() for line in file]
    file.close()
    return brand_list

def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+|\#", "", text)  # Remove mentions & hashtags
    text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

def string2list(string):
    return ast.literal_eval(string)

def get_sentiment_score(tweet):
    return TextBlob(tweet).sentiment.polarity

if __name__ == '__main__':
    df = pd.read_csv('dataset/tweets_data_brute.csv')
    brand_list = read_brand()
    brand2order = {}
    for i in range(len(brand_list)):
        brand2order[brand_list[i]] = i
    df["text"] = df["text"].apply(clean_tweet)
    df["brands"] = df["brands"].apply(string2list)
    df["sentiment_score"] = df["text"].apply(get_sentiment_score)
    df.to_csv('dataset/tweets_data_with_score.csv', index=False)
    