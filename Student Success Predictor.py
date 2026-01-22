import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score, roc_curve,auc, mean_absolute_error, mean_squared_error,confusion_matrix,classification_report
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'study_hours': [5, 3, 8, np.nan, 4, 6, 7, 2, 10, 1],
    'sleep_hours': [7, 6, 5, 8, np.nan, 7, 6, 5, 4, 6],
    'attendance': ['High', 'Low', 'High', 'Medium', 'Medium', 'High', 'High', 'Low', 'High', 'Low'],
    'passed': [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]
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
