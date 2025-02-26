import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from q1 import get_X_y
from q2 import standardize, feature_selection_analysis

if __name__ == '__main__':

    # Load and preprocess Wine Quality dataset
    X, y, feature_names = get_X_y()
    X = standardize(X)
    results = feature_selection_analysis(X, y, feature_names)
    sorted_features = results.index.to_list()[::-1]

    # Define cross-validation
    num_splits = 10
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    train_rmse_dict = {'OLS': [], 'Lasso': [], 'Ridge': []}
    val_rmse_dict = {'OLS': [], 'Lasso': [], 'Ridge': []}

    features_len_list = list(range(1, len(sorted_features)))

    for n in features_len_list:
        selected_features = sorted_features[:n]
        X_selected = X[:, selected_features]

        # Initialize RMSE storage
        train_rmse_sum = {'OLS': 0, 'Lasso': 0, 'Ridge': 0}
        val_rmse_sum = {'OLS': 0, 'Lasso': 0, 'Ridge': 0}

        for train_index, val_index in kf.split(X_selected):
            X_train, X_val = X_selected[train_index], X_selected[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # Train models
            models = {
                'OLS': LinearRegression(),
                'Lasso': Lasso(alpha=0.1, random_state=42),
                'Ridge': Ridge(alpha=1.0, random_state=42)
            }

            for model_name, model in models.items():
                model.fit(X_train, y_train)

                # Predict and calculate RMSE
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)

                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

                train_rmse_sum[model_name] += train_rmse
                val_rmse_sum[model_name] += val_rmse

        # Compute average RMSE
        for model_name in models.keys():
            train_rmse_dict[model_name].append(train_rmse_sum[model_name] / num_splits)
            val_rmse_dict[model_name].append(val_rmse_sum[model_name] / num_splits)

    # Plot results
    plt.figure(figsize=(8, 5))
    for model_name in models.keys():
        plt.plot(features_len_list, train_rmse_dict[model_name], label=f'Train {model_name}')
        plt.plot(features_len_list, val_rmse_dict[model_name], linestyle='dashed', label=f'Val {model_name}')

    plt.xlabel('Number of Features')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison: OLS, Lasso, Ridge Regression')
    plt.legend()
    plt.show()