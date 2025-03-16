import pandas as pd
from sklearn.preprocessing import StandardScaler


def drop_id_fields(df):
    logging.info(f"Dropping ID fields. Before:\n{df.sample(10)}")

    df = df.drop(columns=["Product_ID", "User_ID"], errors="ignore")  # Drop columns safely

    logging.info(f"After dropping ID fields:\n{df.sample(10)}")

    return df

def replace_missing_values_with_unknown(df):
    logging.info(f"Replace missing values with Unknown:\n{df.sample(10)}")

    df["Product_Category_2"].fillna("Unknown", inplace=True)
    df["Product_Category_3"].fillna("Unknown", inplace=True)

    return df
def one_hot_encode(df):
    logging.info(f"One hot encoding:\n{df.sample(10)}")

    df = pd.get_dummies(df, columns=["Gender"], drop_first=True)
    df = pd.get_dummies(df, columns=["Age"], drop_first=True)
    df = pd.get_dummies(df, columns=["Stay_In_Current_City_Years"], drop_first=True)
    df = pd.get_dummies(df, columns=["City_Category"], drop_first=True)
    df = pd.get_dummies(df, columns=["Product_Category_1"], drop_first=True)
    df = pd.get_dummies(df, columns=["Product_Category_2"], drop_first=True)
    df = pd.get_dummies(df, columns=["Product_Category_3"], drop_first=True)


    return df


import logging
def convert_bool_to_int(df):
    # Convert all boolean columns to integers (0 and 1)
    df = df.applymap(lambda x: 1 if x else 0)
    df = df.applymap(lambda x: 1 if x else 0)
    return df

# Feature Engineering - Scaling Features
def scale_features(df):
    logging.info("Feature scaling started...")

    scaler = StandardScaler()

    # Scale only the 'Occupation' column
    df_scaled = df.copy()
    df_scaled["Occupation"] = scaler.fit_transform(df[["Occupation"]])

    logging.info("Feature scaling completed.")
    return df_scaled
