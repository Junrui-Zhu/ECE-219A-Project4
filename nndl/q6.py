import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from q1 import get_X_y
from q2 import standardize, feature_selection_analysis

# Load and standardize the dataset (replace with actual dataset if available)
X, y, feature_names = get_X_y()
X = standardize(X)
results = feature_selection_analysis(X, y, feature_names)
top_features = results.index.to_list()[:5]  # Select top 5 features based on importance
X= X[:, top_features]
# Define hyperparameter search space (no more than 20 experiments)
hidden_layer_sizes_list = [(10,), (20,), (10, 10), (20, 20), (10, 10, 10)]
lr_list = [5e-2, 1e-2, 5e-3, 1e-3]

# Prepare cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = []
experiment_count = 0

# Hyperparameter tuning loop (max 20 experiments)
for hidden_layers in hidden_layer_sizes_list:
    for lr in lr_list:
        if experiment_count >= 20:
            break
        experiment_count += 1
        train_rmse_sum, val_rmse_sum = 0, 0
        
        # Perform 10-fold cross-validation
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Train MLPRegressor
            model = MLPRegressor(hidden_layer_sizes=hidden_layers,
                                 learning_rate_init=lr,
                                 max_iter=500,
                                 random_state=42)
            model.fit(X_train, y_train)

            # Predict and calculate RMSE
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            train_rmse_sum += train_rmse
            val_rmse_sum += val_rmse

        avg_train_rmse = train_rmse_sum / kf.get_n_splits()
        avg_val_rmse = val_rmse_sum / kf.get_n_splits()

        results.append((hidden_layers, lr, avg_train_rmse, avg_val_rmse))

        # Print RMSE for each experiment
        print(f"Experiment {experiment_count}: Hidden layers = {hidden_layers}, lr = {lr}")
        print(f"Average Train RMSE: {avg_train_rmse:.4f}, Average Validation RMSE: {avg_val_rmse:.4f}\n")

# Find best hyperparameter set
best_model = min(results, key=lambda x: x[3])  # Based on validation RMSE
print("Best Model:")
print(f"Hidden layers: {best_model[0]}, Learning rate: {best_model[1]}")
print(f"Train RMSE: {best_model[2]:.4f}, Validation RMSE: {best_model[3]:.4f}")
"""
Question 6.2: How does the performance generally compare with linear regression? Why?
The performance of a multi-layer perceptron (MLP) generally surpasses that of linear regression when dealing with datasets that exhibit non-linear relationships. This is because MLPs, with their multiple hidden layers and non-linear activation functions, can capture complex patterns and interactions between features. In contrast, linear regression assumes a linear relationship between input features and the output, which limits its ability to model more complex data structures. However, if the data is inherently linear, linear regression may perform similarly or even better due to its simplicity and lower risk of overfitting.

Question 6.3: What activation function did you use for the output and why?
The activation function used for the hidden layers is ReLU (Rectified Linear Unit), which is the default in MLPRegressor. ReLU was chosen because it is computationally efficient, helps mitigate the vanishing gradient problem, and accelerates convergence during training. For the output layer, no activation function (i.e., a linear activation) was used. This is appropriate for regression tasks because it allows the model to predict continuous values without constraining the output to a specific range, ensuring flexibility in capturing the target variable’s distribution.

Question 6.4: What is the risk of increasing the depth of the network too far?
Increasing the depth of a neural network too far introduces several risks. Firstly, it can lead to overfitting, where the model becomes overly complex and starts capturing noise in the training data instead of generalizable patterns. Secondly, deeper networks are prone to vanishing or exploding gradient problems during training, which can prevent the model from converging effectively. Additionally, deeper architectures require more computational resources and time for training. Lastly, excessively deep networks can become difficult to interpret, making it challenging to understand how inputs are transformed into outputs.
"""
