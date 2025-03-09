import pandas as pd

import numpy as np

import optuna

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import xgboost as xgb

data = pd.DataFrame({
    'Methanol/Oil Molar Ratio': [5,6,5,5,6,4,6,5,4,4,4,6,4,6,5,6,5,6,5,6,5,5,4,4,4,5,4,4,5,4],
    'Catalyst Weight': [0.7,1.0,0.7,1.0,0.4,1.0,0.4,0.7,1.0,0.4,0.7,1.0,0.7,1.0,0.7,1.0,0.4,0.7,0.7,0.4,0.7,1.0,0.7,1.0,0.4,0.7,1.0,0.4,0.7,0.4],
    'Reaction Temperature': [35,65,50,35,35,65,35,50,65,35,50,60,65,60,50,65,50,50,65,50,50,50,35,65,35,65,65,35,65,65],
    'Reaction Time': [45,30,30,45,30,60,45,45,60,45,60,60,45,60,45,60,45,60,45,30,45,60,30,30,45,45,30,60,45,60],
    'Yield': [93.3,95.88,96.62,94.9,96.4,92.26,96.84,94.22,91.04,84.14,95.9,98.1,98.72,96.86,96.66,95.5,94.58,99.54,95.68,98.41,96.18,95.68,94.86,98.41,85.88,96.86,96.48,94.52,92.26,96.14]
})

X = data[['Methanol/Oil Molar Ratio', 'Catalyst Weight', 'Reaction Temperature', 'Reaction Time']]

y = data['Yield']

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

X_poly = poly.fit_transform(X)

def add_noise(X, y, noise_level=0.02, num_samples=30):
    X_augmented = X.copy()
    y_augmented = y.copy()

    for _ in range(num_samples):  # Generate 30 new samples
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        y_noisy = y + np.random.normal(0, noise_level * np.mean(y), y.shape)

        X_augmented = np.vstack((X_augmented, X_noisy))
        y_augmented = np.concatenate((y_augmented, y_noisy))

    return X_augmented, y_augmented

X_aug, y_aug = add_noise(X_poly, y, noise_level=0.01, num_samples=50)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    return -r2_score(y_test, preds)

study = optuna.create_study(direction='minimize')

study.optimize(objective, n_trials=50)

best_params = study.best_params

import joblib

best_rf_model = RandomForestRegressor(**best_params, random_state=42)

best_rf_model.fit(X_train_scaled, y_train)

xgb_model = xgb.XGBRegressor(n_estimators=500, max_depth=10, learning_rate=0.05, random_state=42)

xgb_model.fit(X_train_scaled, y_train)

rf_preds = best_rf_model.predict(X_test_scaled)


xgb_preds = xgb_model.predict(X_test_scaled)

final_preds = (rf_preds * 0.6) + (xgb_preds * 0.4)

rf_r2 = r2_score(y_test, rf_preds)


xgb_r2 = r2_score(y_test, xgb_preds)


final_r2 = r2_score(y_test, final_preds)


joblib.dump(best_rf_model, "RF_model.pkl")

joblib.dump(xgb_model,"XGB_model.pkl")

print(f"Optimized Random Forest R²: {rf_r2:.4f}")


print(f"XGBoost R²: {xgb_r2:.4f}")


print(f"Final Ensemble R²: {final_r2:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)

plt.scatter(y_test, rf_preds, color='blue', alpha=0.7, label=f'R² = {rf_r2:.4f}')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Ideal")

plt.title("Random Forest - Predicted vs Actual")

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.legend()

plt.subplot(1, 3, 2)

plt.scatter(y_test, xgb_preds, color='green', alpha=0.7, label=f'R² = {xgb_r2:.4f}')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Ideal")

plt.title("XGBoost - Predicted vs Actual")

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.legend()

plt.subplot(1, 3, 3)

plt.scatter(y_test, final_preds, color='orange', alpha=0.7, label=f'R² = {final_r2:.4f}')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Ideal")

plt.title("Final Ensemble - Predicted vs Actual")

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.legend()

plt.tight_layout()

plt.show()