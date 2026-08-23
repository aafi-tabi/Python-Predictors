import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

data_ = {
  "hours_studied": [1, 2, 3, 4, 5, 6],
  "sleep_hours": [5, 6, 7, 6, 8, 7],
  "test_score": [55, 60, 65, 70, 75, 80]
}

data = pd.DataFrame(data_)
X = data[["hours_studied", "sleep_hours"]]
y = data["test_score"]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pip = Pipeline([("scaler",StandardScaler()),
                ("model", LinearRegression())])

pip.fit(X_train,y_train)
y_pred = pip.predict(X_test)

print(f"\n{y_pred}")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test,y_pred)

print(f"\n MAE: {mae}")
print(f"\n MSE: {mse}")

plt.subplot(2,1,1)
plt.plot(X_test["hours_studied"], y_test, linestyle="--", color="skyblue", marker="o", label="actual")
plt.legend()
plt.xlabel("X_test")
plt.ylabel("y_test")
plt.title("actual results")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(X_test["hours_studied"], y_pred, linestyle="--", color="plum", marker="o", label="predicted")
plt.legend()
plt.xlabel("X_test")
plt.ylabel("y_pred")
plt.title("predicted results")
plt.grid(True)

plt.suptitle("comparison of data")
plt.tight_layout()
plt.show()
