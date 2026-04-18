def add_date_features(df):
    df = df.copy()
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month']     = df['date'].dt.month
    df['year']      = df['date'].dt.year
    df['day']       = df['date'].dt.day
    df['quarter']   = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype('int32')
    df['is_month_end']   = df['date'].dt.is_month_end.astype('int32')
    df['dayofyear'] = df['date'].dt.dayofyear
    df['sin_day']   = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_day']   = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['sin_week']  = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_week']  = np.cos(2 * np.pi * df['dayofweek'] / 7)
    return df

def add_lag_features(df):
    grouped = df.groupby(['store_nbr','family'])['sales']
    lags = {15: 'lag_15', 28: 'lag_28', 364: 'lag_364'}
    for shift, col in lags.items():
        df[col] = grouped.shift(shift)
    return df.copy()

def add_payday_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    
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
    df = df.sort_values(['store_nbr', 'family', 'date']).copy()
    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_payday_features(df)
    df = add_rolling_features(df)
    df['oil_lag_1']     = df['dcoilwtico_interpolated'].shift(1)
    df['oil_rolling_7'] = df['dcoilwtico_interpolated'].rolling(7).mean()
    grouped_trans = df.groupby('store_nbr')['transactions']
    df['transactions_lag_15']    = grouped_trans.shift(15)
    df['transactions_rolling_7'] = grouped_trans.transform(lambda x: x.shift(15).rolling(7).mean())
    return df

