import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from models.xgboost_model import XGBoostModel
from helpers import preprocess_appointments

df = pd.read_csv("data/appointments.csv")
X, y = preprocess_appointments(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

model = XGBoostModel()
model.train(X_train_resampled, y_train_resampled)

model.evaluate(X_test, y_test)

model.save_model("models/xgb_model.pkl")