import numpy as np
import lightgbm as lgb
from skopt import BayesSearchCV
from q1 import get_X_y
from q2 import standardize, feature_selection_analysis

if __name__ == "__main__":
    
        # Generate synthetic regression dataset
    X, y, feature_names = get_X_y()
    X = standardize(X)
    results = feature_selection_analysis(X, y, feature_names)
    sorted_features = results.index.to_list()[::-1]
    X_selected = X[:, sorted_features[:7]]
    model = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression', random_state=42)

    # Define the hyperparameter search space
    search_space = {
    'learning_rate': (0.001, 0.1, 'log-uniform'),  
    'num_leaves': (10, 50),        
    'max_depth': (5, 15),          
    'min_data_in_leaf': (10, 50),  
    'feature_fraction': (0.6, 1.0), 
    'bagging_fraction': (0.6, 1.0), 
    'bagging_freq': (1, 5),       
    'lambda_l1': (1e-2, 5),       
    'lambda_l2': (1e-2, 5),       
    'n_estimators': (100, 500),   
}
    # search_space = {
    #     'learning_rate': (0.001, 0.2, 'log-uniform'),  
    #     'num_leaves': (10, 15),  
    #     'max_depth': (-1, 20),  
    #     'min_data_in_leaf': (5, 100),  
    #     'feature_fraction': (0.5, 1.0),  
    #     'bagging_fraction': (0.5, 1.0),  
    #     'bagging_freq': (1, 10),  
    #     'lambda_l1': (1e-3, 10, 'log-uniform'),  
    #     'lambda_l2': (1e-3, 10, 'log-uniform'),  
    #     'n_estimators': (100, 1000),  
    # }

    # Apply Bayesian Optimization with Cross-Validation
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        n_iter=10,  # Number of optimization iterations
        cv=10,  # 10-fold cross-validation
        scoring='neg_root_mean_squared_error',  # Minimize RMSE
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )

    # Train the model with hyperparameter tuning
    opt.fit(X, y)

    # Get the best hyperparameters and model
    best_params = opt.best_params_
    best_model = opt.best_estimator_
    best_rmse = -opt.best_score_

    # Print Results
    print("Best Hyperparameters:", best_params)
    print(f"Cross-Validation RMSE: {best_rmse:.4f}")

