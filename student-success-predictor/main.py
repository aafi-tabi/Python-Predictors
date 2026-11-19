import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score, roc_curve,confusion_matrix,classification_report
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'study_hours': [np.nan, 7, 8, 10, 7, 9, 10, 3, 1, 4, 4, 10, 11, 1, 6, 10, 2, 9, 2, 6, 9, 4, 4, 4, 8, np.nan, 11, 5, 6, 6, 7, 7, 6, np.nan, 9, np.nan, 8, 7, 4, 11],
    'sleep_hours': [6, 5, 8, np.nan, 8, 7, 4, 4, 6, 4, 4, 6, 8, 6, 8, 8, 8, 7, 6, np.nan, 5, 6, 5, 5, 8, 4, 4, 4, 8, np.nan, 8, 5, 7, 5, 6, 4, 7, 8, 7, 4],
    'attendance': ['High', 'Low', 'Medium', 'Low', 'Medium', 'Low', 'High', 'Medium', 'Medium', 'Low', 'High', 'Medium', 'High', 'High', 'High', 'High', 'Low', 'High', 'Low', 'Medium', 'Medium', 'Medium', 'Medium', 'High', 'Medium', 'High', 'Medium', 'High', 'Low', 'High', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Low', 'Low', 'Medium', 'High'],
    'passed': [1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1]
})

X = data.drop("passed", axis=1)
y = data["passed"]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

num_col = ["study_hours","sleep_hours"]
Pipeline_ = Pipeline([("imputer", SimpleImputer(strategy="mean")),
                      ("scalar", StandardScaler())])

cat = ["attendance"]
cat_pip = Pipeline([("encoder", OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([("num",Pipeline_,num_col),
                         ("cat",cat_pip, cat)])

pip = Pipeline([
    ("pipeline", preprocessor),
    ("model", LogisticRegression())])
 
pip.fit(X_train,y_train)
y_pred = pip.predict(X_test)

print(pip.named_steps)
print(f"\n{y_pred}")

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

y_pred_proba = pip.predict_proba(X_test)[:, 1]
print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='red', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
