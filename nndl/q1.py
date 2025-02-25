import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_X_y(folder_path='dataset/'):
    df_red = pd.read_csv(folder_path + 'winequality-red.csv').to_numpy()
    df_white = pd.read_csv(folder_path + 'winequality-white.csv').to_numpy()
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

    return X, y



if __name__ == "__main__":
    X, y = get_X_y()
    print(X.shape, y.shape)


