import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_X_y(folder_path='dataset/'):
    df_red = pd.read_csv(folder_path + 'winequality-red.csv')
    df_white = pd.read_csv(folder_path + 'winequality-white.csv')
    feature_names = df_red.columns[0].replace('"', '').split(';')
    feature_names = ['Type'] + feature_names[:-1]
    df_red = df_red.to_numpy()
    df_white = df_white.to_numpy()
    Red = []
    White = []
    for red in df_red:
        values = red[0].split(';')
        processed_values = [float(value) if '.' in value else int(value) for value in values]
        Red.append(processed_values)
    for white in df_white:
        values = white[0].split(';')
        processed_values = [float(value) if '.' in value else int(value) for value in values]
        White.append(processed_values)
    Type = np.concatenate([np.zeros(len(Red), dtype=int), np.ones(len(White), dtype=int)], axis=0).reshape(-1, 1)
    Red = np.array(Red)
    White = np.array(White)
    Wine = np.concatenate([Red, White], axis=0)
    Wine = np.concatenate([Type, Wine], axis=1)
    X = Wine[:, :-1]
    y = Wine[:, -1]

    return X, y, feature_names

if __name__ == "__main__":

    # q1.1
    X, y, feature_names = get_X_y()
    print(feature_names)
    data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    corr_matrix = np.corrcoef(data, rowvar=False)  # Correlation between columns

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title("Pearson Correlation Heatmap of Wine Datasets")
    plt.show()

    target_corr = corr_matrix[:-1, -1]
    highest_corr_index = np.argmax(np.abs(target_corr))
    highest_corr_value = target_corr[highest_corr_index]

    print(f"Feature with the highest absolute correlation with the target: {feature_names[highest_corr_index]}")
    print(f"Correlation coefficient: {highest_corr_value:.4f}")

    # q1.2
    num_features = len(feature_names)
    plt.figure(figsize=(16, 10))

    for i in range(1, num_features): # discard the first feature 'Type'
        plt.subplot(4, (num_features + 3) // 4, i)
        plt.hist(data[:, i], bins=30, color='skyblue', edgecolor='black')
        plt.title(feature_names[i])

    plt.tight_layout(pad=3.0)
    plt.show()

    # q1.3
    X_winetype = X[:, 0]
    plt.figure(figsize=(8, 6))
    plt.boxplot([y[X_winetype == 0], y[X_winetype == 1]],
                labels=['White', 'Red'], patch_artist=True, 
                boxprops=dict(facecolor='skyblue', color='blue'), 
                medianprops=dict(color='red'))

    plt.title('Box Plot of Quality vs Wine Type')
    plt.xlabel('Category')
    plt.ylabel('Quality')
    plt.show()

    # q1.4
    plt.hist(y, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Quality Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Quality Scores')
    plt.show()