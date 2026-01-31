"""
Feature engineering module for fraud detection system.

This module implements feature extraction from UPI transaction data,
including statistical features, temporal patterns, and behavioral indicators.
Designed for regulatory compliance and bank risk team review.
"""

import pandas as pd
import numpy as np


def build_features(csv_path: str) -> pd.DataFrame:
    """
    Build engineered features from UPI transaction data.
    
    This function processes raw transaction data and creates features suitable
    for machine learning models. Features include:
    - Statistical anomalies (z-scores)
    - Temporal patterns (night transactions)
    - Behavioral indicators (new beneficiaries, transaction velocity)
    - Passthrough features from original data
    
    Args:
        csv_path: Path to the CSV file containing transaction data.
                 Expected columns: txn_id, user_id, amount, txn_hour, is_qr,
                 beneficiary_age_min, device_changed, location_velocity,
                 failed_auth_24h, label
    
    Returns:
        DataFrame with engineered features, ready for ML training.
        Excludes: txn_id, user_id, label (as per requirements)
    """
    # Load the transaction data
    df = pd.read_csv(csv_path)
    
    # Ensure proper data types
    df['user_id'] = df['user_id'].astype(str)
    df['txn_hour'] = df['txn_hour'].astype(int)
    
    # Sort by user_id to ensure proper grouping for rolling calculations
    # Note: In production, you might want to sort by timestamp if available
    df = df.sort_values('user_id').reset_index(drop=True)
    
    # Initialize feature DataFrame with passthrough features
    features = pd.DataFrame({
        'amount': df['amount'].values,
        'is_qr': df['is_qr'].values,
        'device_changed': df['device_changed'].values,
        'location_velocity': df['location_velocity'].values,
        'failed_auth_24h': df['failed_auth_24h'].values,
    })
    
    # Feature 1: amount_zscore (rolling mean/std per user, window=30 transactions)
    # Group by user_id and compute rolling statistics
    df['amount_zscore'] = df.groupby('user_id')['amount'].transform(
        lambda x: _compute_rolling_zscore(x, window=30)
    )
    features['amount_zscore'] = df['amount_zscore'].values
    
    # Feature 2: is_night (1 if txn_hour between 0-4, inclusive)
    features['is_night'] = ((df['txn_hour'] >= 0) & (df['txn_hour'] <= 4)).astype(int)
    
    # Feature 3: beneficiary_is_new (1 if beneficiary_age_min < 10)
    features['beneficiary_is_new'] = (df['beneficiary_age_min'] < 10).astype(int)
    
    # Feature 4: txn_velocity_24h (count of user transactions in last 24 hours)
    # Approximated using rolling window of recent transactions
    # Using window of 30 transactions as approximation for 24-hour activity
    df['txn_velocity_24h'] = df.groupby('user_id')['amount'].transform(
        lambda x: x.rolling(window=30, min_periods=1).count()
    )
    features['txn_velocity_24h'] = df['txn_velocity_24h'].values
    
    # Ensure all features are numeric and handle any NaN values
    features = features.fillna(0)
    
    return features


def _compute_rolling_zscore(series: pd.Series, window: int = 30) -> pd.Series:
    """
    Compute rolling z-score for a series.
    
    Z-score = (value - rolling_mean) / rolling_std
    
    Handles edge cases:
    - If rolling std is zero or missing, z-score is set to 0
    - Uses min_periods=1 to ensure all values have a result
    
    Args:
        series: Input series to compute z-scores for
        window: Rolling window size (default: 30)
    
    Returns:
        Series of z-scores
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    # Compute z-score
    zscore = (series - rolling_mean) / rolling_std
    
    # Handle edge cases: if std is zero or missing, set z-score to 0
    zscore = zscore.fillna(0)
    zscore = zscore.replace([np.inf, -np.inf], 0)
    
    # Additional check: if std is exactly 0, z-score should be 0
    zscore[rolling_std == 0] = 0
    
    return zscore
