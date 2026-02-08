import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv("housing.csv")

X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



models = {
    "Linear Regression": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ]),
        "params": {}
    },

    "Ridge Regression": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge())
        ]),
        "params": {
            "reg__alpha": [0.1, 1, 10, 50]
        }
    },

    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10]
        }
    }
}



best_models = {}

for name, cfg in models.items():

    print("\n🔍 Training:", name)

    grid = GridSearchCV(
        cfg["model"],
        cfg["params"],
        cv=5,
        scoring="r2"
    )

    grid.fit(X_train, y_train)

    best = grid.best_estimator_

    y_pred = best.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Best Params:", grid.best_params_)
    print("MSE:", mse)
    print("R2:", r2)

    best_models[name] = best



print("\n🏆 FINAL MODEL COMPARISON:")

scores = {}

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    scores[name] = r2_score(y_test, y_pred)
    print(name, "R2:", scores[name])

best_model_name = max(scores, key=scores.get)
best_model = best_models[best_model_name]

print("\n✅ BEST MODEL:", best_model_name)



print("\n--- Predict House Price ---")

area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))

user_data = np.array([[area, bedrooms, bathrooms]])

prediction = best_model.predict(user_data)

print("\n💰 Predicted Price:", round(prediction[0], 2))



if best_model_name == "Random Forest":

    importances = best_model.feature_importances_
    features = X.columns

    plt.figure()
    plt.bar(features, importances)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importance in Price Prediction")
    plt.show()




y_test_pred = best_model.predict(X_test)

plt.figure()
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

