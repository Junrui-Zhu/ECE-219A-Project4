from sklearn.feature_selection import mutual_info_regression, f_regression
import numpy as np
import pandas as pd
from q1 import get_X_y
from sklearn.preprocessing import StandardScaler

def standardize(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return X_standardized

def feature_selection_analysis_a(X, y, feature_names):
    mi = mutual_info_regression(X, y)
    f_scores, _ = f_regression(X, y)

    results = pd.DataFrame({
        'Feature': feature_names,
        'Mutual_Information': mi,
        'F_Score': f_scores
    }).sort_values(by='Mutual_Information')
    
    return results

def feature_selection_analysis_d(X, y, feature_names):
    mi = mutual_info_regression(X, y)
    f_scores, _ = f_regression(X, y)

    results = pd.DataFrame({
        'Feature': feature_names,
        'Mutual_Information': mi,
        'F_Score': f_scores
    }).sort_values(by='Mutual_Information', ascending=False)
    
    return results

if __name__ == "__main__":
    X, y, feature_names = get_X_y()
    print(X.shape, y.shape)
    X = standardize(X)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    print("means of each feature\n", means)
    print("variance of each feature:\n", stds)
    results = feature_selection_analysis(X, y, feature_names)
    print("\n", results)
    print("\nlowest 2 MIs:\n", results.head(2))

    """
    To select features that yield better regression results, we employed two feature selection functions from sklearn:

    mutual_info_regression: This function estimates the mutual information (MI) between each feature and the target. MI measures the dependency between variables; higher MI values indicate stronger dependencies.
    f_regression: This function provides F scores, which quantify the significance of each feature in improving the model's predictive performance.
    Qualitative Impact on Test RMSE:
    By removing features with low MI, we reduce noise and redundancy in the dataset, which typically leads to:

    Lower test RMSE due to better generalization.
    Improved convergence speed during training.
    Reduced overfitting, as the model is less likely to learn irrelevant patterns.
    Is This True for All Model Types?

    For linear models (e.g., Linear Regression, Ridge, Lasso), feature selection significantly enhances performance by reducing multicollinearity.
    For non-linear models (e.g., Decision Trees, Random Forests), the impact is less pronounced since these models can inherently handle irrelevant features.
    For deep learning models, the effect depends on the model's complexity and dataset size. Large neural networks often perform internal feature selection implicitly through weight optimization.
    Two Features with the Lowest MI:
    The two features with the lowest mutual information in our dataset are:

    Feature_9 (MI = 0.000952, F_Score = 2.472109)
    Feature_0 (MI = 0.021247, F_Score = 93.811807)
    While Feature_9 is a clear candidate for removal due to its negligible contribution, Feature_0's higher F score suggests that it might still be beneficial for certain models that rely heavily on linear correlations.

    Conclusion:
    Feature selection, guided by mutual information and F scores, qualitatively improves model performance by reducing noise and dimensionality. However, its effectiveness depends on the model type and the nature of the dataset. Experimental validation across multiple models would provide further insights into the universality of these improvements.
    """