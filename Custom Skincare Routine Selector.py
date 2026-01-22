import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

data_ = {
    "skin_type": [
        "Oily", "Dry", "Combo", "Dry", "Oily", "Normal", "Combo", "Oily",
        "Dry", "Sensitive", "Normal", "Combo", "Dry", "Oily", "Sensitive", "Normal",
        "Combo", "Oily", "Dry", "Normal", "Combo", "Dry", "Sensitive", "Oily", "Normal"
    ],
    "age": [
        22, 30, 19, 25, 28, 35, 21, 26,
        29, 31, 24, 20, 27, 23, 33, 36,
        22, 25, 32, 34, 19, 38, 40, 29, 20
    ],
    "routine_type": [
        "Matte", "Hydrating", "Balanced", "Hydrating", "Matte", "Balanced", "Balanced", "Matte",
        "Hydrating", "Hydrating", "Balanced", "Balanced", "Hydrating", "Matte", "Soothing", "Balanced",
        "Balanced", "Matte", "Hydrating", "Balanced", "Balanced", "Hydrating", "Soothing", "Matte", "Balanced"
    ]
}


data = pd.DataFrame(data_)
X = data[["skin_type", "age"]]
y = data["routine_type"]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num = ["age"]
number = Pipeline([("num", StandardScaler())])

cat = ["skin_type"]
cat_ = Pipeline([("cat", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([("num", number, num),
                                  ("cat", cat_, cat)])

model_training = Pipeline([("preprocessor" , preprocessor),
                           ("model", DecisionTreeClassifier())])

model_training.fit(X_train,y_train)

y_pred = model_training.predict(X_test)
print(y_pred)