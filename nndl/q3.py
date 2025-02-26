import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from q1 import get_X_y
from q2 import standardize, feature_selection_analysis

if __name__ == '__main__':

    # Generate synthetic regression dataset
    X, y, feature_names = get_X_y()
    X = standardize(X)
    results = feature_selection_analysis(X, y, feature_names)
    sorted_features = results.index.to_list()[::-1]

    # Define cross-validation
    num_splits = 10
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    train_rmse_list = []
    val_rmse_list = []
    features_len_list = list(range(1, len(sorted_features)))

    for n in features_len_list:
        selected_features = sorted_features[:n]
        X_selected = X[:, selected_features]

        # Store RMSE for training and validation sets
        train_rmse_sum = 0
        val_rmse_sum = 0

        for train_index, val_index in kf.split(X_selected):
            X_train, X_val = X_selected[train_index], X_selected[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # Train model
            model = LogisticRegression(C=0.1, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict and calculate RMSE
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            
            train_rmse_sum += train_rmse
            val_rmse_sum += val_rmse

        # Compute average RMSE
        train_rmse_list.append(train_rmse_sum / num_splits)
        val_rmse_list.append(val_rmse_sum / num_splits)

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(features_len_list, train_rmse_list, label='train')
    plt.plot(features_len_list, val_rmse_list, label='val')
    plt.xlabel('num_features')
    plt.ylabel('RMSE')
    plt.title('Average RMSE under Different Number of Features')
    plt.legend()
    plt.show()