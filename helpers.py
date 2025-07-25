import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_appointments(df):
    # Tarih sütunu kontrolü
    if "scheduling_date" not in df.columns and "scheduled_day" in df.columns:
        df = df.rename(columns={"scheduled_day": "scheduling_date"})

    if "scheduling_date" not in df.columns:
        raise KeyError("Tarih sütunu eksik. 'scheduling_date' veya 'scheduled_day' bekleniyor.")

    df["scheduling_date"] = pd.to_datetime(df["scheduling_date"])
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])

    df["scheduled_days_ahead"] = (df["appointment_date"] - df["scheduling_date"]).dt.days
    df["appointment_dayofweek"] = df["appointment_date"].dt.dayofweek
    df["is_weekend"] = df["appointment_dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

    if "age_group" not in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=[0, 18, 35, 60, 120],
                                 labels=["0-18", "19-35", "36-60", "60+"])

    df["sex_encoded"] = LabelEncoder().fit_transform(df["sex"])
    df["age_group_encoded"] = LabelEncoder().fit_transform(df["age_group"])

    feature_cols = ["scheduled_days_ahead", "appointment_dayofweek", "is_weekend", "sex_encoded", "age_group_encoded"]
    X = df[feature_cols]
    y = df["no_show"] if "no_show" in df.columns else None

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)