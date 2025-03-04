# import json
# import os
# import pandas as pd
# from langdetect import detect
# from tqdm import tqdm
# from rapidfuzz import fuzz

# # 品牌别名字典
# BRAND_ALIASES = {
#     "Budweiser": ["Budweiser"],
#     "Coca-Cola": ["Coca-Cola", "Coke"],
#     "BMW": ["BMW", "Beemer", "Bimmer"],
#     "Audi": ["Audi"],
#     "Mercedes-Benz": ["Mercedes", "Benz", "Mercedes-Benz"],
#     "Microsoft": ["Microsoft"],
#     "Dove": ["Dove"],
#     "Snickers": ["Snickers"],
#     "McDonald's": ["McDonald's", "Mcdonalds", "Mickey D"],
#     "Nike": ["Nike"],
#     "Google": ["Google"],
#     "Lexus": ["Lexus"],
#     "Chevrolet": ["Chevy"],
#     "Dodge": ["Dodge"],
#     "M&M's": ["M&M"],
#     "T-Mobile": ["T-Mobile", "T-Mo"],
#     "Victoria's Secret": ["Victoria's Secret"],
#     "Fiat": ["Fiat"],
#     "Toyota": ["Toyota"],
#     "Pepsi": ["Pepsi"]
# }

# # 计算文本是否包含品牌
# def detect_brands(tweet_text):
#     detected_brands = set()

#     for brand, aliases in BRAND_ALIASES.items():
#         for alias in aliases:
#             similarity = fuzz.partial_ratio(alias.lower(), tweet_text)  # 计算模糊匹配相似度
#             if similarity >= 95:  # 设定相似度阈值
#                 detected_brands.add(brand)
#                 break  # 一个别名匹配成功就跳过

#     return list(detected_brands)

# # 处理单个文件

# def process_tweets(file_path):
#     results = []
#     print(f"Processing file: {file_path}")

#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in tqdm(file, desc="Processing tweets", unit=" tweets"):
#             try:
#                 tweet_data = json.loads(line.strip())
#                 tweet_text = tweet_data.get("tweet", {}).get("text", "").lower()
#                 favorite_count = tweet_data.get("tweet", {}).get("favorite_count", 0)
#                 retweet_count = tweet_data.get("tweet", {}).get("retweet_count", 0)
#                 user = tweet_data.get("tweet", {}).get("")
#                 mentioned_brands = detect_brands(tweet_text)
#                 if mentioned_brands:
#                     results.append({
#                         "text": tweet_text,
#                         "brands": mentioned_brands,
#                         "likes": favorite_count,
#                         "retweets": retweet_count
#                     })

#             except json.JSONDecodeError:
#                 continue  # 跳过无法解析的行

#     return results

# # **主运行逻辑**
# if __name__ == "__main__":
#     folder_path = "D:/ECE219_tweet_data"  # 修改为你的数据文件夹路径

#     tweets_data = []
#     count = []
#     file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]
    
#     for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
#         tweets_data.extend(process_tweets(file_path))

#     # **存储结果**
#     if tweets_data:
#         df = pd.DataFrame(tweets_data)
#         output_csv_path = "D:/ECE219_tweet_data/tweets_data_brute.csv"
#         df.to_csv(output_csv_path, index=False, encoding="utf-8")
#         print(f"Saved results to {output_csv_path} ({len(df)} tweets processed)")
#     else:
#         print("No tweets processed. Check your data files.")
import json
import os
import pandas as pd
import gensim.downloader as api
import gensim
from tqdm import tqdm
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 加载预训练 word2vec 模型
#api.BASE_DIR = "D:/ECE219_tweet_data"
#word_vectors = api.load("word2vec-google-news-300")
model_path = "D:\ECE219_tweet_data\word2vec-google-news-300\word2vec-google-news-300.gz"
word_vectors = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
BRAND_ALIASES = {
    "Budweiser": ["Budweiser"],
    "Coca-Cola": ["Coca-Cola", "Coke"],
    "BMW": ["BMW", "Beemer", "Bimmer"],
    "Audi": ["Audi"],
    "Mercedes-Benz": ["Mercedes", "Benz", "Mercedes-Benz"],
    "Microsoft": ["Microsoft"],
    "Dove": ["Dove"],
    "Snickers": ["Snickers"],
    "McDonald's": ["McDonald's", "Mcdonalds", "Mickey D"],
    "Nike": ["Nike"],
    "Google": ["Google"],
    "Lexus": ["Lexus"],
    "Chevrolet": ["Chevy"],
    "Dodge": ["Dodge"],
    "M&M's": ["M&M"],
    "T-Mobile": ["T-Mobile", "T-Mo"],
    "Victoria's Secret": ["Victoria's Secret"],
    "Fiat": ["Fiat"],
    "Toyota": ["Toyota"],
    "Pepsi": ["Pepsi"],
}

# 计算文本是否包含品牌（基于 word2vec 语义匹配）
def detect_brands(tweet_text):
    detected_brands = set()
    
    tweet_words = tweet_text.lower().split()  # 先拆分推文单词

    for brand, aliases in BRAND_ALIASES.items():
        for alias in aliases:
            if alias.lower() in tweet_text:  # 先检查是否直接包含
                detected_brands.add(brand)
                break

            # 计算语义相似度
            try:
                if alias in word_vectors and any(word in word_vectors for word in tweet_words):
                    tweet_vecs = np.mean([word_vectors[word] for word in tweet_words if word in word_vectors], axis=0)
                    alias_vec = word_vectors[alias]

                    similarity = cosine_similarity([tweet_vecs], [alias_vec])[0][0]  # 计算余弦相似度
                    if similarity > 0.8:  # 设定相似度阈值
                        detected_brands.add(brand)
                        break
            except KeyError:
                continue  # 忽略 OOV（超出词表）的词

    return list(detected_brands)

# 处理单个文件
def process_tweets(file_path):
    results = []
    print(f"Processing file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Processing tweets", unit=" tweets"):
            try:
                tweet_data = json.loads(line.strip())
                tweet_text = tweet_data.get("tweet", {}).get("text", "").lower()
                favorite_count = tweet_data.get("tweet", {}).get("favorite_count", 0)
                retweet_count = tweet_data.get("tweet", {}).get("retweet_count", 0)

                mentioned_brands = detect_brands(tweet_text)
                if mentioned_brands:
                    results.append({
                        "text": tweet_text,
                        "brands": mentioned_brands,
                        "likes": favorite_count,
                        "retweets": retweet_count
                    })

            except json.JSONDecodeError:
                continue  # 跳过无法解析的行

    return results


if __name__ == "__main__":
    folder_path = "D:/ECE219_tweet_data"  # 修改为你的数据文件夹路径

    tweets_data = []
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]

    for file_path in tqdm(file_paths, desc="Processing files", unit="file"):
        tweets_data.extend(process_tweets(file_path))

    # 存储结果
    if tweets_data:
        df = pd.DataFrame(tweets_data)
        output_csv_path = "D:/ECE219_tweet_data/tweets_data.csv"
        df.to_csv(output_csv_path, index=False, encoding="utf-8")
        print(f"Saved results to {output_csv_path} ({len(df)} tweets processed)")
    else:
        print("No tweets processed. Check your data files.")
