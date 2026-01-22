import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer

data_ = {
    "color_score": [0.8, 0.6, 0.3, 0.7, 0.9, 0.4, 0.5, 0.2, 0.95, 0.1],
    "fabric_softness": [0.9, 0.5, 0.2, 0.8, 0.95, 0.3, 0.4, 0.1, 0.99, 0.05],
    "style": ["Kawaii", "Boho", "Street", "Kawaii", "Kawaii", "Boho", "Street", "Street", "Kawaii", "Boho"]
}
data = pd.DataFrame(data_)
X = data[["color_score", "fabric_softness"]]
y = data["style"]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num = ["color_score", "fabric_softness"]
number = Pipeline([("encoder", StandardScaler())])


preprocessor = ColumnTransformer([("num",number,num)
                                  ])

model_training_1 = Pipeline([("preprocessor", preprocessor ),
                           ("model", KNeighborsClassifier(n_neighbors=1))])

model_training_1.fit(X_train,y_train)

y_pred_1 = model_training_1.predict(X_test)
print(f"\n{y_pred_1}")

model_training_2 = Pipeline([("preprocessor", preprocessor ),
                           ("model", KNeighborsClassifier(n_neighbors=2))])


model_training_2.fit(X_train,y_train)

y_pred_2 = model_training_2.predict(X_test)
print(f"\n{y_pred_2}")

model_training_3 = Pipeline([("preprocessor", preprocessor ),
                           ("model", KNeighborsClassifier(n_neighbors=3))])


model_training_3.fit(X_train,y_train)

y_pred_3 = model_training_3.predict(X_test)
print(f"\n{y_pred_3}")


