import pandas as pd
import shutup
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt

shutup.please()

df = pd.read_csv('dataset/tweets_data_with_score.csv')

df['brands'] = df['brands'].apply(lambda x: x.strip("[]").replace("'", "").split(", "))
df = df.explode('brands') 


brand_stats = df.groupby('brands').agg({
    'likes': 'sum',                 
    'retweets': 'sum',             
    'sentiment_score': 'mean'       
}).reset_index()


brand_stats['tweet_count'] = df.groupby('brands').size().values

# normalization
scaler = MinMaxScaler()
features = ['tweet_count', 'likes', 'retweets', 'sentiment_score']
brand_stats_scaled = brand_stats.copy()
brand_stats_scaled[features] = scaler.fit_transform(brand_stats[features])  # 归一化

# clustering
cluster = AgglomerativeClustering(n_clusters=3)
brand_stats_scaled['cluster'] = cluster.fit_predict(brand_stats_scaled[features])  # 预测聚类类别


brand_stats['cluster'] = brand_stats_scaled['cluster']

print(brand_stats.sort_values(by='cluster'))

# visualization
plt.figure(figsize=(12, 7))
sns.scatterplot(
    x=brand_stats['tweet_count'], 
    y=brand_stats['sentiment_score'], 
    hue=brand_stats['cluster'], 
    palette='Set1',
    s=100
)

for i in range(len(brand_stats)):
    if brand_stats['cluster'][i] in [1, 2]:
        plt.text(
            brand_stats['tweet_count'][i], 
            brand_stats['sentiment_score'][i], 
            brand_stats['brands'][i], 
            fontsize=10, 
            ha='right', 
            va='bottom'
        )

plt.xlabel("Tweet Count")
plt.ylabel("Sentiment Score")
plt.title("Brand Clustering based on Tweet Count & Sentiment Score")
plt.legend(title="Cluster")
plt.show()