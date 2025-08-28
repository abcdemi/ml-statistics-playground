# --- Quantile Regression Evaluation Script (Corrected) ---

# 1. SETUP: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Set plot style and figure size
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


# --- METRIC FUNCTION: PINBALL LOSS ---
def pinball_loss(y_true, y_pred, quantile):
    """
    Calculates the pinball loss. This is the correct metric to evaluate quantile forecasts.
    It asymmetrically penalizes errors based on the quantile.
    """
    error = y_true - y_pred
    # The two components of the loss function
    loss_over = np.maximum(error, 0) * quantile
    loss_under = np.maximum(-error, 0) * (1 - quantile)
    return np.mean(loss_over + loss_under)


# --- 2. DATA GENERATION and SPLITTING ---

# Generate heteroscedastic data
np.random.seed(42)
n_samples = 200
X = np.linspace(0, 100, n_samples).reshape(-1, 1)
error_std = 0.1 + 0.05 * X.flatten()
error = np.random.normal(loc=0, scale=error_std)
y = (1.5 * X.flatten() + 10 + error)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# --- 3. MODEL TRAINING (on training data ONLY) ---

# --- Mean-predicting baseline models ---
lr_model = LinearRegression().fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

# --- Quantile Regression Models ---
quantiles = [0.05, 0.5, 0.95]
quantile_models = {}
for q in quantiles:
    qr = QuantileRegressor(quantile=q, alpha=0, solver='highs')
    quantile_models[q] = qr.fit(X_train, y_train)


# --- 4. EVALUATION (on test data ONLY) ---

# --- Calculate metrics for mean-based models ---
rmse_lr = np.sqrt(mean_squared_error(y_test, lr_model.predict(X_test)))
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test)))
rmse_gbr = np.sqrt(mean_squared_error(y_test, gbr_model.predict(X_test)))

# --- Calculate metrics for quantile models ---
pinball_loss_dict = {}
for q, model in quantile_models.items():
    y_pred_q = model.predict(X_test)
    pinball_loss_dict[q] = pinball_loss(y_test, y_pred_q, q)

# Calculate coverage of the 90% prediction interval (between 5th and 95th quantiles)
y_pred_q05 = quantile_models[0.05].predict(X_test)
y_pred_q95 = quantile_models[0.95].predict(X_test)
# Coverage is the percentage of true test points that fall within the interval
coverage = np.mean((y_test >= y_pred_q05) & (y_test <= y_pred_q95)) * 100

# Print results
print("--- Evaluation on Test Set ---")
print(f"Linear Regression RMSE: {rmse_lr:.4f}")
print(f"Random Forest RMSE:     {rmse_rf:.4f}")
print(f"Gradient Boosting RMSE: {rmse_gbr:.4f}\n")
print("--- Quantile Model Evaluation ---")
for q, loss in pinball_loss_dict.items():
    print(f"Pinball Loss for q={q}: {loss:.4f}")
print(f"\nCoverage of 90% Prediction Interval (q=0.05 to q=0.95): {coverage:.2f}%")
print("(An ideal coverage would be 90%)")


# --- 5. VISUALIZATION ---

# Create a smooth line of X values for plotting model predictions
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

# Get predictions from all models
y_plot_lr = lr_model.predict(X_plot)
y_plot_rf = rf_model.predict(X_plot)
y_plot_gbr = gbr_model.predict(X_plot)
y_plot_quantiles = {q: model.predict(X_plot) for q, model in quantile_models.items()}

# Create the plot
plt.figure(figsize=(18, 11))

# Plot training and testing data points differently
plt.scatter(X_train, y_train, color='gray', alpha=0.5, label='Training Data')
plt.scatter(X_test, y_test, color='blue', edgecolor='k', s=80, label='Test Data (for evaluation)')

# Plot the mean-predicting models
plt.plot(X_plot, y_plot_lr, 'r-', linewidth=2, label=f'Linear Regression (Test RMSE: {rmse_lr:.2f})')
plt.plot(X_plot, y_plot_rf, 'g-.', linewidth=2, label=f'Random Forest (Test RMSE: {rmse_rf:.2f})')
# <-- CORRECTED TYPO HERE
plt.plot(X_plot, y_plot_gbr, 'purple', linestyle=(0, (3, 1, 1, 1)), linewidth=2, label=f'Gradient Boosting (Test RMSE: {rmse_gbr:.2f})')

# Plot the quantile regression lines and interval
plt.plot(X_plot, y_plot_quantiles[0.5], 'k-', linewidth=2, label=f'QR Median (Pinball Loss: {pinball_loss_dict[0.5]:.2f})')
plt.plot(X_plot, y_plot_quantiles[0.05], 'k--', linewidth=2, label=f'QR 5th & 95th Percentiles')
plt.plot(X_plot, y_plot_quantiles[0.95], 'k--', linewidth=2)
plt.fill_between(X_plot.flatten(), y_plot_quantiles[0.05], y_plot_quantiles[0.95], alpha=0.2, color='k', label=f'90% Prediction Interval (Coverage: {coverage:.1f}%)')

plt.title('Model Performance on Unseen Test Data', fontsize=20)
plt.xlabel('Predictor Variable (X)', fontsize=14)
plt.ylabel('Target Variable (y)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()