import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Accuracy: {acc}")
        print("Classification Report:")
        print(report)
        return report

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)

    def feature_importance(self, feature_names):
        importances = self.model.feature_importances_
        plt.figure(figsize=(8, 5))
        plt.barh(feature_names, importances)
        plt.xlabel("Importance")
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.savefig("results/feature_importance.png")
        plt.close()