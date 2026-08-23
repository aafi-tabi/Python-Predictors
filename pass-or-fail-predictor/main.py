import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
data_ = {
  "study_hours": [2, 3, 4, 1, 5, 2],
  "class_participation": [1, 0, 1, 0, 1, 0],
  "passed_exam": [1, 1, 1, 0, 1, 0]
}

data = pd.DataFrame(data_)

X = data[["study_hours", "class_participation"]]
y = data["passed_exam"]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_training = Pipeline([("pipeline", StandardScaler()),
                           ("model",LogisticRegression())])

model_training.fit(X_train,y_train)
y_pred = model_training.predict(X_test)
print(y_pred)

y_pred_proba = model_training.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

auc = roc_auc_score(y_test,y_pred_proba)
print(f"\n auc score: {auc}")

plt.plot(fpr, tpr, color="skyblue", marker="o", linestyle="--", label="roc curve")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.legend()
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC CURVE")
plt.grid(True)
plt.show()

