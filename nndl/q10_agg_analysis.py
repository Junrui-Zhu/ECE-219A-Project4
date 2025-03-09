import pandas as pd
import shutup
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt

shutup.please()

# 读取数据集
df = pd.read_csv('dataset/tweets_data_with_score.csv')

# 解析 'brands' 列，去掉 '[]' 和 "'"，然后如果有多个品牌则展开
df['brands'] = df['brands'].apply(lambda x: x.strip("[]").replace("'", "").split(", "))
df = df.explode('brands')  # 处理多个品牌

# 按品牌聚合数据
brand_stats = df.groupby('brands').agg({
    'likes': 'sum',                 # 总点赞数
    'retweets': 'sum',              # 总转发数
    'sentiment_score': 'mean'       # 平均情感得分
}).reset_index()

# 计算每个品牌的推文总数
brand_stats['tweet_count'] = df.groupby('brands').size().values

# **归一化数据**
scaler = MinMaxScaler()
features = ['tweet_count', 'likes', 'retweets', 'sentiment_score']
brand_stats_scaled = brand_stats.copy()
brand_stats_scaled[features] = scaler.fit_transform(brand_stats[features])  # 归一化

# 进行聚类
cluster = AgglomerativeClustering(n_clusters=3)
brand_stats_scaled['cluster'] = cluster.fit_predict(brand_stats_scaled[features])  # 预测聚类类别

# 添加聚类结果回原始数据
brand_stats['cluster'] = brand_stats_scaled['cluster']

# 查看聚类结果
print(brand_stats.sort_values(by='cluster'))

# 可视化：情感得分 vs. 推文数量，按聚类类别着色
plt.figure(figsize=(12, 7))
sns.scatterplot(
    x=brand_stats['tweet_count'], 
    y=brand_stats['sentiment_score'], 
    hue=brand_stats['cluster'], 
    palette='Set1',
    s=100  # 调整点的大小
)

# **添加品牌名称标注（仅标注 cluster=1 和 cluster=2 的点）**
for i in range(len(brand_stats)):
    if brand_stats['cluster'][i] in [1, 2]:  # 只标注第 1、2 类的点
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