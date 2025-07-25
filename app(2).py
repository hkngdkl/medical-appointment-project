import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from models.xgboost_model import XGBoostModel
from helpers import preprocess_appointments
import os

data_path = "data/appointments.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} bulunamadı. Lütfen dosya yolunu kontrol edin.")

appointments_df = pd.read_csv(data_path)

X, y = preprocess_appointments(appointments_df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

model = XGBoostModel()
model.train(X_train_resampled, y_train_resampled)

model.evaluate(X_test, y_test)

model.feature_importance(X.columns)

model.save_model("models/xgb_model.pkl")

print("✅ Model başarıyla eğitildi ve kaydedildi.")