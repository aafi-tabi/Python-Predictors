import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

data_ = {
    "color_score": [0.15, 0.87, 0.77, 0.93, 0.77, 0.2, 0.42, 0.58, 0.73, 0.77, 0.88, 0.77, 0.46, 0.47, 0.94, 0.68, 0.16, 0.32, 0.88, 0.81, 0.59, 0.1, 0.34, 0.37, 0.27, 0.54, 0.6, 0.44, 0.65, 0.2],
    "fabric_softness": [0.19, 0.99, 0.92, 0.83, 0.98, 0.17, 0.05, 0.59, 0.89, 0.77, 0.67, 0.54, 0.1, 0.61, 0.88, 0.54, 0.21, 0.05, 0.92, 0.47, 0.35, 0.61, 0.39, 0.15, 0.42, 0.55, 0.4, 0.75, 0.61, 0.26],
    "style": ['Street', 'Kawaii', 'Kawaii', 'Kawaii', 'Kawaii', 'Street', 'Street', 'Boho', 'Kawaii', 'Kawaii', 'Kawaii', 'Boho', 'Boho', 'Boho', 'Kawaii', 'Kawaii', 'Street', 'Street', 'Kawaii', 'Kawaii', 'Boho', 'Boho', 'Street', 'Boho', 'Boho', 'Boho', 'Boho', 'Boho', 'Kawaii', 'Street']
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
print("Accuracy:", accuracy_score(y_test, y_pred_1))

model_training_2 = Pipeline([("preprocessor", preprocessor ),
                           ("model", KNeighborsClassifier(n_neighbors=2))])


model_training_2.fit(X_train,y_train)

y_pred_2 = model_training_2.predict(X_test)
print(f"\n{y_pred_2}")
print("Accuracy:", accuracy_score(y_test, y_pred_2))

model_training_3 = Pipeline([("preprocessor", preprocessor ),
                           ("model", KNeighborsClassifier(n_neighbors=3))])


model_training_3.fit(X_train,y_train)

y_pred_3 = model_training_3.predict(X_test)
print(f"\n{y_pred_3}")
print("Accuracy:", accuracy_score(y_test, y_pred_3))
