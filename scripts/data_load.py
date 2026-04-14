import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

_REQUIRED_DATA_FILES = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "transactions.csv",
    "oil.csv",
    "holidays_events.csv",
]

_DEFAULT_DATA_DIRS = [
    Path("data/store-sales-time-series-forecasting"),
    Path("data"),
]

OUTPUT_DIR = Path(os.environ.get("P5_OUTPUT_DIR", "answers"))
_RESOLVED_DATA_DIR = None


def _unique_paths(paths):
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path)
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def data_dir_candidates():
    candidates = []
    env_dir = os.environ.get("P5_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend(_DEFAULT_DATA_DIRS)
    return _unique_paths(candidates)


def resolve_data_dir():
    global _RESOLVED_DATA_DIR

    if _RESOLVED_DATA_DIR is not None:
        return _RESOLVED_DATA_DIR

    for base in data_dir_candidates():
        if all((base / file_name).exists() for file_name in _REQUIRED_DATA_FILES):
            _RESOLVED_DATA_DIR = base
            return _RESOLVED_DATA_DIR

    data_root = Path("data")
    if data_root.exists():
        for train_path in data_root.rglob("train.csv"):
            base = train_path.parent
            if all((base / file_name).exists() for file_name in _REQUIRED_DATA_FILES):
                _RESOLVED_DATA_DIR = base
                return _RESOLVED_DATA_DIR

    searched = "\n  - ".join(str(path) for path in data_dir_candidates())
    raise FileNotFoundError(
        "Could not locate the P5 dataset locally.\n"
        "Expected the extracted CSV files in one of these directories:\n"
        f"  - {searched}\n"
        "You can also set the environment variable P5_DATA_DIR to the extracted dataset directory."
    )


def competition_file(file_name):
    return resolve_data_dir() / file_name



def output_file(file_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / file_name



def data_import():
    train = pd.read_csv(competition_file("train.csv"))
    test = pd.read_csv(competition_file("test.csv"))
    stores = pd.read_csv(competition_file("stores.csv"))
    transactions = pd.read_csv(competition_file("transactions.csv")).sort_values(["store_nbr", "date"])
    oil = pd.read_csv(competition_file("oil.csv"))
    holidays = pd.read_csv(competition_file("holidays_events.csv"))

    # Datetime
    train["date"] = pd.to_datetime(train.date)
    test["date"] = pd.to_datetime(test.date)
    transactions["date"] = pd.to_datetime(transactions.date)
    oil["date"] = pd.to_datetime(oil.date)
    holidays["date"] = pd.to_datetime(holidays.date)

    # Data types
    train.onpromotion = train.onpromotion.astype("float16")
    train.sales = train.sales.astype("float32")
    stores.cluster = stores.cluster.astype("int8")

    return train, test, stores, transactions, oil, holidays



def make_submission(test_preds, file_name="submission.csv"):
    """
    Args:
        test_preds: Your predictions from the model.
        file_name: Output file name. The file is written to the local answers/ directory.

    NOTE: Your test predictions should be in the same order as the test set.
          This function does not take care of unsorted/shuffled predictions.
    """
    test = pd.read_csv(competition_file("test.csv"))
    submission_df = pd.DataFrame(columns=["id", "sales"])
    submission_df.sales = test_preds
    submission_df.id = test.id.astype(int)
    submission_df.to_csv(output_file(file_name), index=False)



def rmsle(y_pred, y_true):
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))



def lgbm_rmsle(preds, train_data):
    labels = train_data.get_label()
    rmsle_val = rmsle(preds, labels)
    return 'RMSLE', rmsle_val, False



def preprocess_holidays(holidays):
    # # Process holidays and events
    tr1 = holidays[(holidays.type == "Holiday") & (holidays.transferred == True)].drop("transferred", axis=1).reset_index(drop=True)
    tr2 = holidays[(holidays.type == "Transfer")].drop("transferred", axis=1).reset_index(drop=True)
    tr = pd.concat([tr1, tr2], axis=1)
    tr = tr.iloc[:, [5, 1, 2, 3, 4]]

    holidays = holidays[(holidays.transferred == False) & (holidays.type != "Transfer")].drop("transferred", axis=1)
    holidays = pd.concat([holidays, tr], ignore_index=True).reset_index(drop=True)

    # Additional Holidays
    holidays["description"] = holidays["description"].str.replace("-", "").str.replace("+", "").str.replace(r'\d+', '', regex=True)
    holidays["type"] = np.where(holidays["type"] == "Additional", "Holiday", holidays["type"])

    # # Bridge Holidays
    holidays["description"] = holidays["description"].str.replace("Puente ", "")
    holidays["type"] = np.where(holidays["type"] == "Bridge", "Holiday", holidays["type"])

    # # Work Day Holidays, that is meant to payback the Bridge.
    work_day = holidays[holidays.type == "Work Day"]
    holidays = holidays[holidays.type != "Work Day"]

    # # Events are national
    events = holidays[holidays.type == "Event"].drop(["type", "locale", "locale_name"], axis=1).rename({"description": "events"}, axis=1)
    holidays = holidays[holidays.type != "Event"].drop("type", axis=1)
    regional = holidays[holidays.locale == "Regional"].rename({"locale_name": "state", "description": "holiday_regional"}, axis=1).drop("locale", axis=1).drop_duplicates()
    national = holidays[holidays.locale == "National"].rename({"description": "holiday_national"}, axis=1).drop(["locale", "locale_name"], axis=1).drop_duplicates()
    local = holidays[holidays.locale == "Local"].rename({"description": "holiday_local", "locale_name": "city"}, axis=1).drop("locale", axis=1).drop_duplicates()
    events["events"] = np.where(events.events.str.contains("futbol"), "Futbol", events.events)

    return holidays, regional, national, local, events, work_day, tr, tr1, tr2



def preprocess_test_train(merged_df, one_hot_encoder, stores):
    holidays = pd.read_csv(competition_file("holidays_events.csv"))
    holidays["date"] = pd.to_datetime(holidays.date)
    holidays, regional, national, local, events, work_day, tr, tr1, tr2 = preprocess_holidays(holidays)

    d = pd.merge(merged_df, stores)
    d["store_nbr"] = d["store_nbr"].astype("int8")

    # National Holidays & Events
    d = pd.merge(d, national, how="left")
    # Regional
    d = pd.merge(d, regional, how="left", on=["date", "state"])
    # Local
    d = pd.merge(d, local, how="left", on=["date", "city"])

    # Work Day: It will be removed when real work day column created
    d = pd.merge(d, work_day[["date", "type"]].rename({"type": "IsWorkDay"}, axis=1), how="left")

    events, events_cat = one_hot_encoder(events, nan_as_category=False)
    events["events_Dia_de_la_Madre"] = np.where(events.date == "2016-05-08", 1, events["events_Dia_de_la_Madre"])
    events = events.drop(239)

    d = pd.merge(d, events, how="left")
    d[events_cat] = d[events_cat].fillna(0)

    # New features
    d["holiday_national_binary"] = np.where(d.holiday_national.notnull(), 1, 0)
    d["holiday_local_binary"] = np.where(d.holiday_local.notnull(), 1, 0)
    d["holiday_regional_binary"] = np.where(d.holiday_regional.notnull(), 1, 0)
    d["national_independence"] = np.where(d.holiday_national.isin(['Batalla de Pichincha', 'Independencia de Cuenca', 'Independencia de Guayaquil', 'Independencia de Guayaquil', 'Primer Grito de Independencia']), 1, 0)
    d["local_cantonizacio"] = np.where(d.holiday_local.str.contains("Cantonizacio"), 1, 0)
    d["local_fundacion"] = np.where(d.holiday_local.str.contains("Fundacion"), 1, 0)
    d["local_independencia"] = np.where(d.holiday_local.str.contains("Independencia"), 1, 0)

    holidays, holidays_cat = one_hot_encoder(d[["holiday_national", "holiday_regional", "holiday_local"]], nan_as_category=False)
    d = pd.concat([d.drop(["holiday_national", "holiday_regional", "holiday_local"], axis=1), holidays], axis=1)

    he_cols = (
        d.columns[d.columns.str.startswith("events")].tolist()
        + d.columns[d.columns.str.startswith("holiday")].tolist()
        + d.columns[d.columns.str.startswith("national")].tolist()
        + d.columns[d.columns.str.startswith("local")].tolist()
    )
    d[he_cols] = d[he_cols].astype("int8")

    d[["family", "city", "state", "type"]] = d[["family", "city", "state", "type"]].astype("category")

    del holidays, holidays_cat, work_day, local, regional, national, events, events_cat, tr, tr1, tr2, he_cols

    train = d[d.date < "2017-08-01"]
    test = d[d.date >= "2017-08-01"]
    test = test.drop(["sales"], axis=1)

    # After one-hot encoding is applied to both datasets
    train = pd.get_dummies(train, columns=train.select_dtypes(include=['object', 'str']).columns)
    train = pd.get_dummies(train, columns=train.select_dtypes(['category']).columns)

    test = pd.get_dummies(test, columns=test.select_dtypes(include=['object', 'str']).columns)
    test = pd.get_dummies(test, columns=test.select_dtypes(['category']).columns)

    # Add missing columns to test (set to 0)
    missing_cols = set(train.columns) - set(test.columns) - {'sales'}
    for col in missing_cols:
        test[col] = 0

    # Ensure test has the same column order as train (except 'sales')
    train_cols = [col for col in train.columns if col != 'sales']
    test = test[train_cols]

    for col in train.columns:
        if pd.api.types.is_numeric_dtype(train[col]):
            train[col] = train[col].astype('float32')

    for col in test.columns:
        if pd.api.types.is_numeric_dtype(test[col]):
            test[col] = test[col].astype('float32')

    replace_dict = {}
    for x in train.columns:
        if "family" in x:
            if " " in x:
                lst = x.split(" ")
                new_feature_name = ""
                for k in lst:
                    new_feature_name += str(k)
                    new_feature_name += "_"
                replace_dict[x] = new_feature_name[:-1]
            if "," in x:
                lst = x.split(",")
                new_feature_name = ""
                for k in lst:
                    new_feature_name += str(k)
                    new_feature_name += "_"
                replace_dict[x] = new_feature_name[:-1]
            if "/" in x:
                lst = x.split(",")
                new_feature_name = ""
                for k in lst:
                    new_feature_name += str(k)
                    new_feature_name += "_"
                replace_dict[x] = new_feature_name[:-1]

        elif "state" in x:
            lst = x.split(",")
            new_feature_name = ""
            for k in lst:
                new_feature_name += str(k)
                new_feature_name += "_"
            replace_dict[x] = new_feature_name[:-1]

    train.rename(columns=replace_dict)
    test.rename(columns=replace_dict)

    train = train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    test = test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    return train, test
