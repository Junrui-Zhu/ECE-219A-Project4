import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from q1 import get_X_y
from q2 import standardize, feature_selection_analysis_d

if __name__ == '__main__':
    # Load and preprocess the dataset
    X, y, feature_names = get_X_y()
    X = standardize(X)  # Standardize features

    # Perform feature selection analysis (Ensure deterministic selection)
    np.random.seed(42)  # Fix random seed for reproducibility
    results = feature_selection_analysis_d(X, y, feature_names)
    
    # ✅ Ensure exactly 7 features are selected
    top_feature_indices = list(results.index[:7])  # Extract only the top 7 feature indices
    top_feature_names = [feature_names[i] for i in top_feature_indices]  # Get correct feature names
    X_selected = X[:, top_feature_indices]  # Select only those 7 features

    # ✅ Print selected features (without intercept)
    print("\nSelected Top 7 Features:")
    for i, feature in enumerate(top_feature_names, start=1):
        print(f"{i}. {feature}")

    # Double-check the number of selected features
    assert len(top_feature_names) == 7, f"Error: Expected 7 features, but got {len(top_feature_names)}"
    assert X_selected.shape[1] == 7, f"Error: Expected X_selected to have 7 columns, but got {X_selected.shape[1]}"

    # Set up 10-fold cross-validation (Ensure deterministic splits)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Initialize RMSE storage
    train_rmse_dict = {'OLS': [], 'Lasso': [], 'Ridge': []}
    val_rmse_dict = {'OLS': [], 'Lasso': [], 'Ridge': []}

    # Train models using 10-fold cross-validation
    models = {
        'OLS': LinearRegression(),
        'Lasso': Lasso(alpha=0.1, random_state=42),
        'Ridge': Ridge(alpha=1.0, random_state=42)
    }

    for train_index, val_index in kf.split(X_selected):
        X_train, X_val = X_selected[train_index], X_selected[val_index]
        y_train, y_val = y[train_index], y[val_index]

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

    # Print RMSE results
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

    # Compute p-values using OLS regression
    X_selected = sm.add_constant(X_selected)  # Add intercept for regression
    model_ols = sm.OLS(y, X_selected).fit()
    p_values = model_ols.pvalues[1:]  # Exclude intercept

    # Store and print feature p-values (excluding intercept)
    feature_p_values = dict(zip(top_feature_names, p_values))

    print("\nP-values for the Top 7 Features:")
    for feature, p_value in feature_p_values.items():
        significance = "Significant" if p_value < 0.05 else "Not Significant"
        print(f"{feature}: p-value = {p_value:.4f}  ({significance})")