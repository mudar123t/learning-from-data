from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

def calculate_correlation(df, target_column):
    """
    Calculates correlation of all features with the target column.
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): Column name for the target variable.

    Returns:
        pd.Series: Correlation values sorted by absolute value.
    """
    numerical_df = df.select_dtypes(include=[np.number])
    correlations = numerical_df.corr()[target_column].sort_values(key=abs, ascending=False)

    return correlations

def select_features_mrmr(X, y, k=10):
    """
    Selects top-k features using Minimum Redundancy Maximum Relevance (mRMR).
    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target variable.
        k (int): Number of features to select.

    Returns:
        list: Selected feature names.
    """
    scaler = StandardScaler()
    X = X.fillna(0)  # Replace missing values with zero or another appropriate strategy
    X_scaled = scaler.fit_transform(X)
    mi = mutual_info_regression(X_scaled, y)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    selected_features = mi_series.head(k).index.tolist()
    return selected_features
