import logging

import pandas as pd

from black_friday_feature_engg import drop_id_fields


def load_dataset():
    df = pd.read_csv("data/black_friday.csv")
    logging.info("Dataset loaded ...")

    return df

def save_dataset(df):
    df.to_csv("data/black_friday_transformed_data.csv", index=False)
    logging.info("Dataset loaded ...")

    return df, df.columns

def load_transformed_dataset():
    df = pd.read_csv("data/black_friday_transformed_data.csv")
    logging.info("Dataset loaded ...")
    df = drop_id_fields(df)
    return df
