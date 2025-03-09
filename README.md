# Project 3: Recommender Systems

## Table of Contents
1. [How to Run the Code](#how-to-run-the-code)
2. [Dependencies](#dependencies)
3. [File Structure](#file-structure)
4. [Authors](#authors)
---
## How to Run the Code
Necessary steps to run the code:

1. Navigate to the project directory:
   ```bash
   cd ECE-219A-Project4
   ```

2. Install the required dependencies. (Check #dependencies for required packages and version)

3. There are 2 original datasets in this project, TXT files of twitter data and wine quality dataset. In our submitted codes, only 1 dataset (wine) is included. As for the twitter data, we uploaded the pre-processed version of it, named as tweets_data_xxx.csv. However, you do need to download the original dataset and change the file path accordingly to run q10_data_fetch.py successfully. All uploaded data files are in the /dataset directory.

4. Each python script performs a certain algorithm/task, or answers several questions. For question1-9, they are answered by q1-9.py, respectively. As for question 10, we created q10_data_fetch.py to select and generate twitters of interest, q_10_sentiment_analysis.py to perform sentiment analysis and add sentiment score for selected twitter data, q10_agg_analysis.py performs agglomerative clustering for all selected brands. All python scripts are in /nndl directory. For more detailed information of how and why we process/select data in such way, please refer to our report.
---

## Dependencies
This project requires the following libraries and tools. 
- python==3.11.8
- pandas==1.5.3
- numpy==1.24.0
- matplotlib==3.6.3
- seaborn==0.11.2
- scikit-learn==1.1.3
- scipy==1.10.1
- nltk==3.8.1
- textblob==0.15.3
- langdetect==1.0.9
- scikit-learn>=1.2.0
- tqdm==4.64.0
- shutup==0.1.3
- statsmodels==0.14.0
- scikit-optimize==0.9.0
- lightgbm==4.1.0
- rapidfuzz==3.5.2
---

## File Structure
This Project is organized as follows:
```bash  
├── dataset/   
│   ├── brand.txt
│   ├── tweets_data_brute.csv
│   ├── tweets_data_with_score.csv
│   ├── tweets_data.csv 
│   ├── wineequality-red.csv 
│   └── wineequality-white.csv         
├── nndl/                 # Source code
│   ├── q1.py      
│   ├── q2.py
│   ├── q3.py
│   ├── q4.py
│   ├── q5.py
│   ├── q6.py
│   ├── q7.py
│   ├── q8.py
│   ├── q9.py   
│   ├── q10_agg_analysis.py
│   ├── q10_data_fetch.py
│   └── q10_sentiment_analysis.py 
└── README.md            # Documentation
```

Notes:
- All source code is located in the `nndl/` folder.
---

## Authors

This project was collaboratively developed by the following contributors:

| Name                | UID                       |  Contact               |
|---------------------|---------------------------|------------------------|
| **LiWei Tan**       | 206530851                 | 962439602@qq.com       |
| **TianXiang Xing**  | 006530550                 | andrewxing43@g.ucla.edu|
| **Junrui Zhu**      | 606530444                 | zhujr24@g.ucla.edu     |
---
