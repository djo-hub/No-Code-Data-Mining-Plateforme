import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def train_model(df, features, target, algorithm, task_type, test_size=0.2, params=None):
    """
    Train machine learning model based on task type and algorithm

    Args:
        df: pandas DataFrame
        features: list of feature column names
        target: target column name (None for clustering)
        algorithm: algorithm name
        task_type: 'Classification', 'Regression', or 'Clustering'
        test_size: test set size for supervised learning
        params: additional parameters for clustering

    Returns:
        model and data splits
    """
    if task_type in ["Classification", "Regression"]:
        # Prepare data
        X = df[features].dropna()
        y = df.loc[X.index, target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Select and train model
        if task_type == "Classification":
            if algorithm == "K-Nearest Neighbors":
                model = KNeighborsClassifier(n_neighbors=5)
            elif algorithm == "Support Vector Machine":
                model = SVC(kernel='rbf', random_state=42)
            elif algorithm == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42, max_depth=5)
            elif algorithm == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)

        else:  # Regression
            if algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm == "Support Vector Regression":
                model = SVR(kernel='rbf')
            elif algorithm == "Decision Tree Regression":
                model = DecisionTreeRegressor(random_state=42, max_depth=5)

        # Train model
        model.fit(X_train_scaled, y_train)

        return model, X_test_scaled, X_test, y_train, y_test

    else:  # Clustering
        # Prepare data
        X = df[features].dropna()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select and train clustering model
        if algorithm == "K-Means":
            model = KMeans(n_clusters=params['n_clusters'], random_state=42, n_init=10)
        elif algorithm == "Hierarchical Clustering":
            model = AgglomerativeClustering(n_clusters=params['n_clusters'])
        elif algorithm == "DBSCAN":
            model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])

        # Fit and predict
        labels = model.fit_predict(X_scaled)

        return model, X_scaled, labels

def evaluate_model(model, X_test, y_test, task_type):
    """
    Evaluate model performance

    Args:
        model: trained model
        X_test: test features
        y_test: test targets
        task_type: 'Classification' or 'Regression'

    Returns:
        dict with evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, r2_score, mean_absolute_error
    )

    y_pred = model.predict(X_test)

    if task_type == "Classification":
        # Handle multi-class vs binary
        average_method = 'weighted'

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=average_method, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=average_method, zero_division=0),
            'f1': f1_score(y_test, y_pred, average=average_method, zero_division=0)
        }

    else:  # Regression
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }

    return metrics
