import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from q1 import get_X_y
from q2 import standardize, feature_selection_analysis

if __name__ == '__main__':

    # Load and preprocess Wine Quality dataset
    X, y, feature_names = get_X_y()
    X = standardize(X)
    
    # Perform feature selection analysis
    results = feature_selection_analysis(X, y, feature_names)
    top_features = results.index.to_list()[:5]  # Select top 5 features for polynomial regression
    X_selected = X[:, top_features]

    # Define cross-validation
    num_splits = 10
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

    degrees = range(1, 7)  # Polynomial degrees from 1 to 6
    train_rmse_list = []
    val_rmse_list = []

    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_selected)  # Transform features to polynomial terms

        train_rmse_sum = 0
        val_rmse_sum = 0

        for train_index, val_index in kf.split(X_poly):
            X_train, X_val = X_poly[train_index], X_poly[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # Train Ridge regression on polynomial features
            model = Ridge(alpha=1.0, random_state=42)
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
    plt.plot(degrees, train_rmse_list, label='Train RMSE', marker='o')
    plt.plot(degrees, val_rmse_list, label='Validation RMSE', marker='s')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('RMSE')
    plt.title('Polynomial Regression Performance')
    plt.legend()
    plt.show()