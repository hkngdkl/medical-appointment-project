import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from helpers import preprocess_appointments
from models.xgboost_model import XGBoostModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/appointments.csv")
X, y = preprocess_appointments(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = XGBoostModel()
model.train(X_train_res, y_train_res)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

with open("results/classification_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)

plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Attended", "No-Show"], yticklabels=["Attended", "No-Show"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results/roc_curve.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.barh(X.columns, model.model.feature_importances_)
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
plt.close()