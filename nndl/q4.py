import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from q1 import get_X_y
from q2 import standardize, feature_selection_analysis_d  # Corrected function to get the best 5 features

if __name__ == '__main__':

    # Load and preprocess the Wine Quality dataset
    X, y, feature_names = get_X_y()
    X = standardize(X)  # Standardize features to have mean 0 and variance 1

    # Perform feature selection analysis and select the top 5 features
    results = feature_selection_analysis_d(X, y, feature_names)  # Corrected to get the best features
    top_features = results.index.to_list()[:7]  # Select top 7 features based on importance
    X_selected = X[:, top_features]  # Extract selected features

    # Set up 10-fold cross-validation
    num_splits = 10
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Initialize dictionaries to store RMSE results
    train_rmse_dict = {'OLS': [], 'Lasso': [], 'Ridge': []}
    val_rmse_dict = {'OLS': [], 'Lasso': [], 'Ridge': []}

    # Train models using only the top 5 selected features
    for train_index, val_index in kf.split(X_selected):
        X_train, X_val = X_selected[train_index], X_selected[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Define models
        models = {
            'OLS': LinearRegression(),
            'Lasso': Lasso(alpha=0.1, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42)
        }

        # Train each model and compute RMSE
        for model_name, model in models.items():
            model.fit(X_train, y_train)

            # Predict and calculate RMSE
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

            train_rmse_dict[model_name].append(train_rmse)
            val_rmse_dict[model_name].append(val_rmse)

    # Compute average RMSE for each model
    avg_train_rmse = {model: np.mean(train_rmse_dict[model]) for model in models.keys()}
    avg_val_rmse = {model: np.mean(val_rmse_dict[model]) for model in models.keys()}

    # Print results
    for model in models.keys():
        print(f"{model}: Train RMSE = {avg_train_rmse[model]:.4f}, Validation RMSE = {avg_val_rmse[model]:.4f}")

    # Plot RMSE comparison
    plt.figure(figsize=(8, 5))
    plt.bar(models.keys(), avg_train_rmse.values(), alpha=0.6, label='Train RMSE')
    plt.bar(models.keys(), avg_val_rmse.values(), alpha=0.6, label='Validation RMSE')
    plt.ylabel('RMSE')
    plt.title('Linear Regression Performance (Top 7 Features)')
    plt.legend()
    plt.show()