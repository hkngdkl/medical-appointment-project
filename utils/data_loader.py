import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)

    df["scheduling_date"] = pd.to_datetime(df["scheduling_date"])
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    df["scheduled_days_ahead"] = (df["appointment_date"] - df["scheduling_date"]).dt.days
    df["appointment_dayofweek"] = df["appointment_date"].dt.dayofweek
    df["is_weekend"] = df["appointment_dayofweek"].isin([5, 6]).astype(int)

    df = df[df["scheduled_days_ahead"] >= 0]

    df["sex_encoded"] = df["sex"].map({"Female": 0, "Male": 1})
    
    age_group_mapping = {label: idx for idx, label in enumerate(sorted(df["age_group"].unique()))}
    df["age_group_encoded"] = df["age_group"].map(age_group_mapping)

    features = ["scheduled_days_ahead", "appointment_dayofweek", "is_weekend", "sex_encoded", "age_group_encoded"]
    X = df[features]
    y = df["no_show"]

    return df, X, y