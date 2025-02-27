import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from q1 import get_X_y
from q2 import standardize, feature_selection_analysis_d

if __name__ == '__main__':

    # Load and preprocess the Wine Quality dataset
    X, y, feature_names = get_X_y()
    X = standardize(X)  # Standardize features to have mean 0 and variance 1
    
    # Perform feature selection analysis and select the top 5 features
    results = feature_selection_analysis_d(X, y, feature_names)
    top_features = results.index.to_list()[:5]  # Select top 5 features based on importance
    X_selected = X[:, top_features]  # Extract selected features

    # Set up 10-fold cross-validation
    num_splits = 10
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    # Define a range of polynomial degrees to test
    degrees = range(1, 7)  # Polynomial degrees from 1 to 6
    train_rmse_list = []  # To store average training RMSE for each degree
    val_rmse_list = []    # To store average validation RMSE for each degree

    # Iterate over different polynomial degrees
    for degree in degrees:
        # Generate polynomial features of the specified degree
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_selected)  # Transform features into polynomial terms

        train_rmse_sum = 0  # Accumulate training RMSE over folds
        val_rmse_sum = 0    # Accumulate validation RMSE over folds

        # Perform cross-validation
        for train_index, val_index in kf.split(X_poly):
            X_train, X_val = X_poly[train_index], X_poly[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Train Ridge regression model on polynomial features
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions and calculate RMSE for both training and validation sets
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

            train_rmse_sum += train_rmse
            val_rmse_sum += val_rmse

        # Compute average RMSE over all folds
        avg_train_rmse = train_rmse_sum / num_splits
        avg_val_rmse = val_rmse_sum / num_splits

        train_rmse_list.append(avg_train_rmse)
        val_rmse_list.append(avg_val_rmse)

        print(f"Polynomial degree {degree}: Train RMSE = {avg_train_rmse:.4f}, Validation RMSE = {avg_val_rmse:.4f}")

    # Plot RMSE results for training and validation sets
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_rmse_list, label='Train RMSE', marker='o')
    plt.plot(degrees, val_rmse_list, label='Validation RMSE', marker='s')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('RMSE')
    plt.title('Polynomial Regression Performance')
    plt.legend()
    plt.grid(True)
    plt.show()
