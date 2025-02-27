import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
from q1 import get_X_y
from q2 import standardize, feature_selection_analysis_d

# Load and standardize the dataset
X, y, feature_names = get_X_y()
X = standardize(X)
results = feature_selection_analysis_d(X, y, feature_names)
top_features = results["Feature"].to_list()[:5]  # Select top 5 features based on importance
print(top_features)
# Define Random Forest hyperparameters
n_estimators = 100  # Number of trees
max_depth = 4       # Maximum depth of each tree
max_features = 'sqrt'  # Maximum number of features considered at each split

# Prepare 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
train_rmse_list = []
val_rmse_list = []
r2_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train Random Forest Regressor with OOB estimation
    model = RandomForestRegressor(n_estimators=n_estimators, 
                                  max_depth=max_depth,
                                  max_features=max_features,
                                  oob_score=True,
                                  random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and calculate RMSE and R2 score
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    
    train_rmse_list.append(train_rmse)
    val_rmse_list.append(val_rmse)
    r2_scores.append(r2)

# Compute average scores
avg_train_rmse = np.mean(train_rmse_list)
avg_val_rmse = np.mean(val_rmse_list)
avg_r2 = np.mean(r2_scores)

oob_error = 1 - model.oob_score_  # Compute Out-of-Bag Error

# Print results
print(f"Average Train RMSE: {avg_train_rmse:.4f}")
print(f"Average Validation RMSE: {avg_val_rmse:.4f}")
print(f"Average R2 Score: {avg_r2:.4f}")
print(f"Out-of-Bag Error: {oob_error:.4f}")

# Plot feature importance
feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(10, 5))
sns.barplot(x=np.array(feature_names)[sorted_indices], y=feature_importances[sorted_indices])
plt.xticks(rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance in Random Forest")
plt.show()

# Visualize a single tree in the random forest
plt.figure(figsize=(12, 8))
plot_tree(model.estimators_[0], feature_names=feature_names, filled=True, rounded=True, max_depth=4)
plt.title("Visualization of a Single Tree (Max Depth = 4)")
plt.show()
