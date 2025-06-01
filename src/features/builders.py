import pandas as pd
import numpy as np

def add_time_features(df, datetime_column):
    """
    Adds time-based features to the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        datetime_column (str): Column name containing datetime values.

    Returns:
        pd.DataFrame: DataFrame with added time features.
    """
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df['year'] = df[datetime_column].dt.year
    df['month'] = df[datetime_column].dt.month
    df['day'] = df[datetime_column].dt.day
    df['hour'] = df[datetime_column].dt.hour
    df['day_of_week'] = df[datetime_column].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Cyclical encoding for time-based features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

def add_rolling_features(df, column, windows=[3, 6, 12]):
    """
    Adds rolling statistics (mean and std) for a given column.
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): Column name to calculate rolling statistics for.
        windows (list): List of window sizes.

    Returns:
        pd.DataFrame: DataFrame with added rolling statistics.
    """
    for window in windows:
        df[f'{column}_rolling_mean_{window}h'] = df[column].rolling(window=window).mean()
        df[f'{column}_rolling_std_{window}h'] = df[column].rolling(window=window).std()

    return df

def add_lag_features(df, column, lags=[1, 24, 48]):
    """
    Adds lag features for a given column.
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): Column name to create lag features for.
        lags (list): List of lag values.

    Returns:
        pd.DataFrame: DataFrame with added lag features.
    """
    df = df.dropna(subset=[f'{column}_lag_{lag}h' for lag in lags])

    return df

def add_weather_features(df, temperature_column, wind_speed_column):
    """
    Adds derived weather features such as wind chill.
    Args:
        df (pd.DataFrame): The input DataFrame.
        temperature_column (str): Column name for temperature data.
        wind_speed_column (str): Column name for wind speed data.

    Returns:
        pd.DataFrame: DataFrame with added weather features.
    """
    # Wind chill calculation
    df['wind_chill'] = 13.12 + 0.6215 * df[temperature_column] - 11.37 * (df[wind_speed_column] ** 0.16) + 0.3965 * df[temperature_column] * (df[wind_speed_column] ** 0.16)
    
    # Apparent temperature as an example
    df['apparent_temp'] = df[temperature_column] - (0.7 * df[wind_speed_column])

    return df
