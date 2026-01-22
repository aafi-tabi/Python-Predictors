import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

data_ = {
    "lines_of_code": [100, 250, 300, 150, 400, 500, 120, 80, 700, 320, 180, 600, 290, 270, 360, 420, 510, 130, 50, 100],
    "bugs_fixed":     [2,   5,   7,   3,   10,  12,  1,   0,  20,  6,   2,   15,  4,   5,   8,   11,  13,  2,  0,  1],
    "build_success":  [1,   1,   0,   1,   1,   1,   0,  0,  1,   0,   1,   1,   0,   1,   0,   1,   1,  0,  0,  0]
}


data = pd.DataFrame(data_)
X = data[["lines_of_code","bugs_fixed"]]
y = data["build_success"]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2,  random_state=42)

model_training = Pipeline([("scaler", StandardScaler()),
                          ("model", RandomForestClassifier())])

model_training.fit(X_train,y_train)
y_pred = model_training.predict(X_test)

print(y_pred)