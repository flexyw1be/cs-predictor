import pandas as pd
from config import DATA_PATH


def load_data():
    df = pd.read_csv(DATA_PATH)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)

    print(f"Data loaded. Shape: {df.shape}")
    return df


def split_time_series_data(X, y, split_ratio=0.8):
    split_index = int(len(X) * split_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test