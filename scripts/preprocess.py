import re

import numpy as np
import pandas as pd

from scripts.data import competition_file
from scripts.holidays import preprocess_holidays


def oil_preprocess(oil):
    oil_daily = oil.set_index("date").resample("D").asfreq()
    oil_daily["dcoilwtico"] = oil_daily["dcoilwtico"].replace(0, np.nan)
    oil_daily["dcoilwtico_interpolated"] = oil_daily["dcoilwtico"].interpolate()
    return oil_daily


def build_raw_merged(train, test, transactions, oil):
    oil_daily = oil_preprocess(oil)
    return (
        pd.concat([train, test])
        .merge(transactions, how="left")
        .merge(oil_daily.reset_index(), how="left", on="date")
        .drop(columns=["dcoilwtico"])
    )


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = df.select_dtypes(["category", "object"]).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    rename_map = {col: col.replace(" ", "_") for col in df.columns if col not in original_columns}
    df = df.rename(columns=rename_map)
    new_columns = list(rename_map.values())
    return df, new_columns


def preprocess_test_train(merged_df, holidays_df, stores):
    holidays, regional, national, local, events, work_day = preprocess_holidays(holidays_df)

    d = pd.merge(merged_df, stores)
    d["store_nbr"] = d["store_nbr"].astype("int8")

    d = pd.merge(d, national, how="left")
    d = pd.merge(d, regional, how="left", on=["date", "state"])
    d = pd.merge(d, local, how="left", on=["date", "city"])
    d = pd.merge(d, work_day[["date", "type"]].rename(columns={"type": "IsWorkDay"}), how="left")

    events, events_cat = one_hot_encoder(events, nan_as_category=False)
    # Fix Mother's Day 2016 mis-labeled row and drop the duplicate
    events["events_Dia_de_la_Madre"] = np.where(events.date == "2016-05-08", 1, events["events_Dia_de_la_Madre"])
    events = events.drop(index=239, errors="ignore")

    d = pd.merge(d, events, how="left")
    d[events_cat] = d[events_cat].fillna(0)

    # Binary holiday presence flags (before one-hot encoding drops the text columns)
    d["holiday_national_binary"] = d.holiday_national.notnull().astype("int8")
    d["holiday_local_binary"]    = d.holiday_local.notnull().astype("int8")
    d["holiday_regional_binary"] = d.holiday_regional.notnull().astype("int8")

    # Grouped holiday sub-types
    independence_holidays = [
        "Batalla de Pichincha", "Independencia de Cuenca",
        "Independencia de Guayaquil", "Primer Grito de Independencia",
    ]
    d["national_independence"] = d.holiday_national.isin(independence_holidays).astype("int8")
    d["local_cantonizacio"]    = d.holiday_local.str.contains("Cantonizacio", na=False).astype("int8")
    d["local_fundacion"]       = d.holiday_local.str.contains("Fundacion", na=False).astype("int8")
    d["local_independencia"]   = d.holiday_local.str.contains("Independencia", na=False).astype("int8")

    holidays_ohe, holidays_cat = one_hot_encoder(
        d[["holiday_national", "holiday_regional", "holiday_local"]], nan_as_category=False
    )
    d = pd.concat([d.drop(["holiday_national", "holiday_regional", "holiday_local"], axis=1), holidays_ohe], axis=1)

    he_cols = [c for c in d.columns if c.startswith(("events", "holiday", "national", "local"))]
    d[he_cols] = d[he_cols].astype("int8")

    d[["family", "city", "state", "type"]] = d[["family", "city", "state", "type"]].astype("category")

    del holidays, holidays_cat, work_day, local, regional, national, events, events_cat, he_cols

    train = d[d.date < "2017-08-01"].copy()
    test  = d[d.date >= "2017-08-01"].drop(columns=["sales"]).copy()

    # One-hot encode remaining categoricals
    train = pd.get_dummies(train, columns=train.select_dtypes(["object", "category"]).columns)
    test  = pd.get_dummies(test,  columns=test.select_dtypes(["object", "category"]).columns)

    # Align test columns to train (fill any train-only columns with 0)
    for col in set(train.columns) - set(test.columns) - {"sales"}:
        test[col] = 0
    test = test[[col for col in train.columns if col != "sales"]]

    # Cast all numeric columns to float32
    for df in [train, test]:
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].astype("float32")

    # Sanitize column names: strip all non-alphanumeric/underscore characters
    clean = lambda x: re.sub(r"[^A-Za-z0-9_]+", "", x)
    train = train.rename(columns=clean)
    test  = test.rename(columns=clean)

    return train, test
