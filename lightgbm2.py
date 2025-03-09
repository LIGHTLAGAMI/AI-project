import pandas as pd
import numpy as np
import optuna
import joblib
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data
data = pd.DataFrame({
    'Methanol/Oil Molar Ratio': [5,6,5,5,6,4,6,5,4,4,4,6,4,6,5,6,5,6,5,6,5,5,4,4,4,5,4,4,5,4],
    'Catalyst Weight': [0.7,1.0,0.7,1.0,0.4,1.0,0.4,0.7,1.0,0.4,0.7,1.0,0.7,1.0,0.7,1.0,0.4,0.7,0.7,0.4,0.7,1.0,0.7,1.0,0.4,0.7,1.0,0.4,0.7,0.4],
    'Reaction Temperature': [35,65,50,35,35,65,35,50,65,35,50,60,65,60,50,65,50,50,65,50,50,50,35,65,35,65,65,35,65,65],
    'Reaction Time': [45,30,30,45,30,60,45,45,60,45,60,60,45,60,45,60,45,60,45,30,45,60,30,30,45,45,30,60,45,60],
    'Yield': [93.3,95.88,96.62,94.9,96.4,92.26,96.84,94.22,91.04,84.14,95.9,98.1,98.72,96.86,96.66,95.5,94.58,99.54,95.68,98.41,96.18,95.68,94.86,98.41,85.88,96.86,96.48,94.52,92.26,96.14]
})

# Features & Target
X = data.drop(columns=['Yield'])
y = data['Yield']

# Add Polynomial Features (Interaction Terms)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# Data Augmentation (Noise Injection)
def add_noise(X, y, noise_level=0.01, num_samples=30):
    X_augmented = X.copy()
    y_augmented = y.copy()

    for _ in range(num_samples):
        noise = np.random.normal(0, noise_level, X.shape)
        X_noisy = X + noise
        y_noisy = y + np.random.normal(0, noise_level * np.mean(y), y.shape)

        X_augmented = np.vstack((X_augmented, X_noisy))
        y_augmented = np.concatenate((y_augmented, y_noisy))

    return X_augmented, y_augmented

X_aug, y_aug = add_noise(X_poly, y, noise_level=0.01, num_samples=50)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning using Optuna
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    return -r2_score(y_test, preds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Train Optimized LightGBM Model
best_params = study.best_params
lgb_model = lgb.LGBMRegressor(**best_params)
lgb_model.fit(X_train_scaled, y_train)
joblib.dump(lgb_model,"lightgbm_model")
# Predictions
y_pred = lgb_model.predict(X_test_scaled)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Optimized LightGBM MAE: {mae:.4f}")
print(f"Optimized LightGBM RMSE: {rmse:.4f}")
print(f"Optimized LightGBM R² Score: {r2:.4f}")
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', label=f'Predicted vs Actual (R² = {r2:.4f})')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed', linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Optimized LightGBM Regression: Actual vs Predicted Yield")
plt.legend()
plt.show()
