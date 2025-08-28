# --- Quantile Regression Demonstration Script ---

# 1. SETUP: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Set plot style and figure size
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 9)

# --- 2. DATA GENERATION with HETEROSCEDASTICITY ---

# We will create data where the variance of y increases as x increases.
# This is a classic use case where modeling the mean is not enough.
np.random.seed(42)
n_samples = 200

# Create the predictor variable
X = np.linspace(0, 100, n_samples).reshape(-1, 1)

# The error term's standard deviation will depend on X, creating the "fan" shape.
# This is the key to heteroscedasticity.
error_std = 0.1 + 0.05 * X.flatten()
error = np.random.normal(loc=0, scale=error_std)

# Create the target variable with a base linear trend
y = (1.5 * X.flatten() + 10 + error)

# Add a few outliers to demonstrate robustness
X = np.append(X, [[20], [80]]).reshape(-1, 1)
y = np.append(y, [120, 10])


# --- 3. MODEL TRAINING ---

# Baseline Model 1: Ordinary Least Squares (OLS) Linear Regression
# This model finds the conditional mean.
lr_model = LinearRegression()
lr_model.fit(X, y)

# Baseline Model 2: Random Forest Regressor
# A powerful model that also targets the conditional mean.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Baseline Model 3: Gradient Boosting Regressor
# Another powerful model targeting the conditional mean.
gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr_model.fit(X, y)

# --- Quantile Regression Models ---
# We will model three quantiles to capture the changing distribution.
quantiles = [0.05, 0.5, 0.95]
quantile_models = {}

for q in quantiles:
    # The `solver='highs'` is a modern, efficient solver available in recent scikit-learn versions
    # For older versions, you might need to remove this argument.
    qr = QuantileRegressor(quantile=q, alpha=0, solver='highs')
    quantile_models[q] = qr.fit(X, y)


# --- 4. VISUALIZATION AND COMPARISON ---

# Create a smooth line of X values for plotting our model predictions
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

# Get predictions from all models
y_pred_lr = lr_model.predict(X_plot)
y_pred_rf = rf_model.predict(X_plot)
y_pred_gbr = gbr_model.predict(X_plot)
y_pred_quantiles = {q: model.predict(X_plot) for q, model in quantile_models.items()}


# Create the plot
plt.figure(figsize=(16, 10))

# Plot the raw data points
plt.scatter(X, y, color='gray', alpha=0.6, label='Data Points')

# Plot the mean-predicting models
plt.plot(X_plot, y_pred_lr, 'r-', linewidth=2, label='Linear Regression (Mean)')
plt.plot(X_plot, y_pred_rf, 'g-.', linewidth=2, label='Random Forest (Mean)')
plt.plot(X_plot, y_pred_gbr, 'b-.', linewidth=2, label='Gradient Boosting (Mean)')

# Plot the quantile regression lines
plt.plot(X_plot, y_pred_quantiles[0.5], 'k-', linewidth=2, label='Quantile Regression (Median, q=0.5)')
plt.plot(X_plot, y_pred_quantiles[0.05], 'k--', linewidth=2, label='Quantile Regression (q=0.05)')
plt.plot(X_plot, y_pred_quantiles[0.95], 'k--', linewidth=2, label='Quantile Regression (q=0.95)')

# Fill the area between the upper and lower quantiles to represent the prediction interval
plt.fill_between(X_plot.flatten(), y_pred_quantiles[0.05], y_pred_quantiles[0.95], alpha=0.2, color='k', label='90% Prediction Interval')


plt.title('Quantile Regression vs. Mean Regression on Heteroscedastic Data', fontsize=18)
plt.xlabel('Predictor Variable (X)', fontsize=14)
plt.ylabel('Target Variable (y)', fontsize=14)
plt.legend(fontsize=12)
plt.show()