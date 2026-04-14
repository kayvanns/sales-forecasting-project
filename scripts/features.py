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
    grouped = df.groupby(['store_nbr','family')['sales']
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

def build_features(df):
    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_payday_features(df)
    return df

def one_hot_encoder(df, nan_as_category=True):
    # One hot encode all object/category columns.
    # Leave non-categorical columns unchanged.
    # If nan_as_category=True, include missing-value dummy columns.
    # Replace spaces in output column names with underscores.
    # Return the encoded dataframe and the list of newly created columns in output order.

    original_columns = list(df.columns)
    
    categorical_columns = df.select_dtypes(["category", "object", "str"]).columns.tolist()
    
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    
    rename_map = {
        col: col.replace(" ", "_")
        for col in df.columns
        if col not in original_columns
    }
    
    df = df.rename(columns=rename_map)
    new_columns = [rename_map[col] for col in rename_map]

    return df, new_columns
