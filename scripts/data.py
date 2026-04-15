import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/store-sales-time-series-forecasting")
OUTPUT_DIR = Path("submissions")

def competition_file(file_name):
    return DATA_DIR / file_name

def output_file(file_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / file_name


def make_submission(test_preds, file_name="submission.csv"):
    test = pd.read_csv(competition_file("test.csv"))
    submission_df = pd.DataFrame(columns=["id", "sales"])
    submission_df.sales = test_preds
    submission_df.id = test.id.astype(int)
    submission_df.to_csv(output_file(file_name), index=False)


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


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))


def lgbm_rmsle(preds, train_data):
    labels = train_data.get_label()
    rmsle_val = rmsle(labels, preds)
    return 'RMSLE', rmsle_val, False





