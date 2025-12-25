import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def explore_data(df):
    """
    Generate exploration statistics for the dataset

    Args:
        df: pandas DataFrame

    Returns:
        dict with exploration information
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum()
    }
    return info

def preprocess_data(df, operation, method=None):
    """
    Preprocess data based on selected operation

    Args:
        df: pandas DataFrame
        operation: 'missing', 'duplicates', or 'all'
        method: method for handling missing values

    Returns:
        processed DataFrame
    """
    df_processed = df.copy()

    if operation in ['missing', 'all']:
        if method == "Drop rows with missing values" or method == "drop":
            df_processed = df_processed.dropna()
        elif method == "Fill with mean":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        elif method == "Fill with median":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        elif method == "Fill with mode":
            for col in df_processed.columns:
                if not df_processed[col].mode().empty:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    if operation in ['duplicates', 'all']:
        df_processed = df_processed.drop_duplicates()

    return df_processed

def encode_categorical(df, columns):
    """
    Encode categorical variables

    Args:
        df: pandas DataFrame
        columns: list of column names to encode

    Returns:
        encoded DataFrame and encoders dict
    """
    df_encoded = df.copy()
    encoders = {}

    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df_encoded, encoders
