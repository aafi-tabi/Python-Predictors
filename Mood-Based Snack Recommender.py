from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

features = [["happy","red"],["sad","blue"],["calm","green"],["excited","red"],["tired","purple"]]


snacks = ["donut","chocolate","matcha cake","candy","coffee"]

ohe = OneHotEncoder()
X = ohe.fit_transform(features).toarray()

le = LabelEncoder()
y = le.fit_transform(snacks)

model = DecisionTreeClassifier()
model.fit(X,y)

new_input = [["happy", "red"]]
X_test = ohe.transform(new_input)

predict = model.predict(X_test)
predict_output = le.inverse_transform(predict)

print("Your snack suggestion:", predict_output[0])




