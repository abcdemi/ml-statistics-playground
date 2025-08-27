# --- Skewed Data Handling ---

# 1. SETUP: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer

# Set plot style and figure size
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
pd.options.mode.chained_assignment = None # Hide warning

# --- UTILITY FUNCTION FOR PLOTTING ---
def plot_distributions(original_data, transformed_data_dict, title_prefix):
    """Plots the original distribution against several transformed distributions."""
    num_plots = len(transformed_data_dict) + 1
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))

    # Plot original
    sns.histplot(original_data, kde=True, ax=axes[0])
    axes[0].set_title(f"Original (Skew: {original_data.skew():.2f})")

    # Plot transformed
    i = 1
    for name, data in transformed_data_dict.items():
        sns.histplot(data, kde=True, ax=axes[i])
        axes[i].set_title(f"{name} (Skew: {data.skew():.2f})")
        i += 1
    fig.suptitle(f"{title_prefix}: Original vs. Transformed Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==============================================================================
# --- EXPERIMENT 1: HANDLING RIGHT-SKEWED DATA ---
# ==============================================================================

print("="*50)
print("EXPERIMENT 1: Handling Right-Skewed Data")
print("="*50)

# --- 1.1 Generate Right-Skewed Data ---
# Common in finance, real estate prices, etc.
np.random.seed(42)
X_right = np.linspace(1, 10, 200)
# Create a non-linear relationship and make y right-skewed by exponentiating
noise = np.random.normal(0, 0.2, 200)
y_right_skew = np.exp(0.4 * X_right + noise) + 10

data_right = pd.DataFrame({'X': X_right, 'y': y_right_skew})

# --- 1.2 Initial Visualization ---
print(f"Initial Skewness of y: {data_right['y'].skew():.4f}")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data_right['y'], kde=True, ax=ax1)
ax1.set_title("Distribution of Right-Skewed Target (y)")
sns.scatterplot(x='X', y='y', data=data_right, ax=ax2)
ax2.set_title("Feature vs. Target Relationship")
plt.show()

# --- 1.3 Apply Transformations for Right-Skew ---
transformed_right = {}
# Log Transformation: Compresses large values. Use log1p for safety (handles y=0).
transformed_right['Log (log1p)'] = np.log1p(data_right['y'])
# Square Root Transformation: Milder than log.
transformed_right['Square Root'] = np.sqrt(data_right['y'])
# Cube Root Transformation
transformed_right['Cube Root'] = np.cbrt(data_right['y'])
# Box-Cox Transformation: Finds the optimal power transformation (lambda). Requires data > 0.
y_boxcox, boxcox_lambda = stats.boxcox(data_right['y'])
transformed_right['Box-Cox'] = y_boxcox
print(f"Box-Cox optimal lambda: {boxcox_lambda:.4f}")
# Yeo-Johnson Transformation: Handles data with zeros and negative values.
pt = PowerTransformer(method='yeo-johnson')
y_yeojohnson = pt.fit_transform(data_right[['y']]).flatten()
transformed_right['Yeo-Johnson'] = y_yeojohnson

# Visualize the effect of transformations
plot_distributions(data_right['y'], transformed_right, "Right-Skew")

# --- 1.4 Model Comparison ---
X_train, X_test, y_train, y_test = train_test_split(data_right['X'], data_right['y'], test_size=0.3, random_state=42)
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

# --- Simple Model: Linear Regression ---
# a) On raw, skewed data
lr_raw = LinearRegression().fit(X_train, y_train)
y_pred_raw_lr = lr_raw.predict(X_test)
rmse_raw_lr = np.sqrt(mean_squared_error(y_test, y_pred_raw_lr))

# b) On log-transformed data (best transformation for this case)
# We transform y, fit the model, predict, and then inverse-transform the predictions
y_train_log = np.log1p(y_train)
lr_log = LinearRegression().fit(X_train, y_train_log)
y_pred_log_lr = lr_log.predict(X_test)
# CRITICAL: Inverse transform predictions to compare RMSE on the original scale
y_pred_untransformed_lr = np.expm1(y_pred_log_lr)
rmse_log_lr = np.sqrt(mean_squared_error(y_test, y_pred_untransformed_lr))

# --- Advanced Model: Random Forest ---
# Tree-based models are less sensitive to feature skew, but target skew can still be an issue.
# a) On raw, skewed data
rf_raw = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred_raw_rf = rf_raw.predict(X_test)
rmse_raw_rf = np.sqrt(mean_squared_error(y_test, y_pred_raw_rf))

# b) On log-transformed data
rf_log = RandomForestRegressor(random_state=42).fit(X_train, y_train_log)
y_pred_log_rf = rf_log.predict(X_test)
y_pred_untransformed_rf = np.expm1(y_pred_log_rf)
rmse_log_rf = np.sqrt(mean_squared_error(y_test, y_pred_untransformed_rf))

# --- 1.5 Results & Visualization ---
results = {
    "Linear Regression (Raw)": rmse_raw_lr,
    "Linear Regression (Log Transformed)": rmse_log_lr,
    "Random Forest (Raw)": rmse_raw_rf,
    "Random Forest (Log Transformed)": rmse_log_rf,
}
print("\n--- Model Performance on Right-Skewed Data ---")
for name, rmse in results.items():
    print(f"{name}: Test RMSE = {rmse:.4f}")

plt.figure(figsize=(12, 7))
plt.scatter(X_test, y_test, alpha=0.6, label='Actual Test Data')
plt.plot(np.sort(X_test.flatten()), np.sort(y_pred_raw_lr), 'r--', label='LR (Raw) Prediction')
plt.plot(np.sort(X_test.flatten()), np.sort(y_pred_untransformed_lr), 'r-', linewidth=2, label='LR (Transformed) Prediction')
plt.plot(np.sort(X_test.flatten()), np.sort(y_pred_raw_rf), 'g--', label='RF (Raw) Prediction')
plt.plot(np.sort(X_test.flatten()), np.sort(y_pred_untransformed_rf), 'g-', linewidth=2, label='RF (Transformed) Prediction')
plt.title("Model Predictions on Raw vs. Transformed Data")
plt.legend()
plt.show()

# ==============================================================================
# --- EXPERIMENT 2: HANDLING LEFT-SKEWED DATA ---
# ==============================================================================
print("\n" + "="*50)
print("EXPERIMENT 2: Handling Left-Skewed Data")
print("="*50)

# --- 2.1 Generate Left-Skewed Data ---
np.random.seed(42)
y_left_skew = 100 - stats.lognorm(s=0.6, scale=30).rvs(200)
print(f"Initial Skewness of y: {pd.Series(y_left_skew).skew():.4f}")

# --- 2.2 Initial Visualization ---
sns.histplot(y_left_skew, kde=True)
plt.title("Distribution of Left-Skewed Data")
plt.show()

# --- 2.3 Apply Transformations for Left-Skew ---
transformed_left = {}
# For left-skew, we reflect the data to be right-skewed, transform it, then reflect it back.
# A simpler approach shown here is to use powers.
# Square Transformation: Can work for moderate left skew.
transformed_left['Square'] = y_left_skew ** 2
# Cube Transformation: Stronger transformation.
transformed_left['Cube'] = y_left_skew ** 3
# Yeo-Johnson Transformation: A robust method that works here.
pt_left = PowerTransformer(method='yeo-johnson')
y_yeojohnson_left = pt_left.fit_transform(y_left_skew.reshape(-1, 1)).flatten()
transformed_left['Yeo-Johnson'] = y_yeojohnson_left

# Visualize transformations
plot_distributions(pd.Series(y_left_skew), transformed_left, "Left-Skew")