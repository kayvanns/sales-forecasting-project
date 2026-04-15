import pandas as pd

def add_date_features(df):
    date_features = pd.concat([
        df["date"].dt.dayofweek.rename("dayofweek"),
        df["date"].dt.month.rename("month"),
        df["date"].dt.year.rename("year"),
        df["date"].dt.day.rename("day"),
        df["date"].dt.quarter.rename("quarter"),
        df["date"].dt.is_month_start.rename("is_month_start").astype("int32"),
        df["date"].dt.is_month_end.rename("is_month_end").astype("int32"),
        
    ], axis=1)
    return pd.concat([df, date_features], axis=1)

def add_lag_features(df):
    grouped = df.groupby(['store_nbr','family'])['sales']
    lags = {15: 'lag_15', 28: 'lag_28', 364: 'lag_364'}
    for shift, col in lags.items():
        df[col] = grouped.shift(shift)
    return df.copy()

def add_payday_features(df):
    df = df.copy()
    
    def days_to_next_payday(date):
        day = date.day
        last = date.days_in_month
        if day < 15:
            return 15 - day
        elif day < last:
            return last - day
        else:
            return 0  
    
    def days_since_last_payday(date):
        day = date.day
        last = date.days_in_month
        if day >= last:
            return 0 
        elif day > 15:
            return day - 15
        elif day == 15:
            return 0  
        else:
            return day  
    
    df["days_to_payday"] = df["date"].apply(days_to_next_payday)
    df["days_since_payday"] = df["date"].apply(days_since_last_payday)
    return df

def add_rolling_features(df):
    grouped = df.groupby(['store_nbr', 'family'])['sales']
    df['rolling_mean_7']  = grouped.transform(lambda x: x.shift(15).rolling(7).mean())
    df['rolling_mean_28'] = grouped.transform(lambda x: x.shift(15).rolling(28).mean())
    df['rolling_std_7']   = grouped.transform(lambda x: x.shift(15).rolling(7).std())
    return df
    
def build_features(df):
    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_payday_features(df)
    df = add_rolling_features(df)
    return df
