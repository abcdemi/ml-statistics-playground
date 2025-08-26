# --- Outlier Handling Demonstration Script (Corrected & Improved) ---

# 1. SETUP: Import libraries and create a synthetic dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Set plot style and figure size
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# --- Data Generation ---
# Create a base dataset with a clear linear relationship
np.random.seed(42)
X_clean = np.linspace(0, 10, 100).reshape(-1, 1)
y_clean = 2.5 * X_clean.flatten() + 5 + np.random.normal(0, 2, 100)

# Introduce a few significant outliers
X_outliers = np.append(X_clean, np.array([[1], [2], [9]])).reshape(-1, 1)
y_outliers = np.append(y_clean, np.array([35, 40, 0])) # Add 3 points far from the trend

# Create a DataFrame for easier manipulation
data = pd.DataFrame(X_outliers, columns=['X'])
data['y'] = y_outliers


# --- 2. BASELINE MODEL & IMPROVED OUTLIER DETECTION ---

# Fit a model on the data with outliers to see the negative effect
model_baseline = LinearRegression()
model_baseline.fit(data[['X']], data['y'])
y_pred_baseline = model_baseline.predict(data[['X']])

# Visualize the baseline model
plt.scatter(data['X'], data['y'], color='blue', alpha=0.6, label='Data Points')
plt.plot(data['X'], y_pred_baseline, color='red', linewidth=2, label='Baseline Model (Poor Fit)')
plt.title('Effect of Outliers on Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# --- Outlier Detection using Residuals and IQR (The Correct Way for Regression) ---
# An outlier in regression is a point that deviates from the fitted line.
# We detect this by finding points with large errors (residuals).
data['residuals'] = data['y'] - y_pred_baseline

# Math: IQR = Q3 - Q1. Apply this to the residuals now.
# An outlier is a point whose residual falls outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
Q1 = data['residuals'].quantile(0.25)
Q3 = data['residuals'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outlier indices based on residuals
outlier_indices = data[(data['residuals'] < lower_bound) | (data['residuals'] > upper_bound)].index
print(f"--- Outlier Detection using Residuals & IQR ---\nLower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
print(f"Outliers detected at indices: {outlier_indices.tolist()}\n") # This will now work!

# Visualize detection with a boxplot of the residuals
plt.figure(figsize=(8, 6))
sns.boxplot(y=data['residuals'])
plt.title('Boxplot of Model Residuals for Outlier Detection')
plt.ylabel('Residual (Error)')
plt.show()


# --- 3. METHODS FOR HANDLING OUTLIERS ---

results = {} # Dictionary to store model performance

# Store baseline results
results['Baseline'] = {
    'coef': model_baseline.coef_[0],
    'intercept': model_baseline.intercept_,
    'r2': r2_score(data['y'], y_pred_baseline),
    'rmse': np.sqrt(mean_squared_error(data['y'], y_pred_baseline))
}

# --- Method 1: Correction (Simulated) ---
# We simulate fixing a "data entry error". Let's assume the outlier y=0 should have been y=25.
data_corrected = data.copy()
# We will use the identified index. Let's assume index 102 corresponds to the y=0 point.
data_corrected.loc[102, 'y'] = 25.0
model_corrected = LinearRegression().fit(data_corrected[['X']], data_corrected['y'])
results['Correction'] = {
    'coef': model_corrected.coef_[0],
    'intercept': model_corrected.intercept_,
    'r2': r2_score(data_corrected['y'], model_corrected.predict(data_corrected[['X']])),
    'rmse': np.sqrt(mean_squared_error(data_corrected['y'], model_corrected.predict(data_corrected[['X']])))
}

# --- Method 2: Removal ---
# We remove the rows identified as outliers. This is the most common approach.
data_removed = data.drop(outlier_indices)
model_removed = LinearRegression().fit(data_removed[['X']], data_removed['y'])
results['Removal'] = {
    'coef': model_removed.coef_[0],
    'intercept': model_removed.intercept_,
    'r2': r2_score(data_removed['y'], model_removed.predict(data_removed[['X']])),
    'rmse': np.sqrt(mean_squared_error(data_removed['y'], model_removed.predict(data_removed[['X']])))
}

# --- Method 3: Transformation ---
# Apply a log transformation to compress the y-values and reduce the outlier's influence.
data_transformed = data.copy()
# Note: Transformation is less effective for outliers that are "low" (like y=0)
data_transformed['y_log'] = np.log1p(data_transformed['y'])
model_transformed = LinearRegression().fit(data_transformed[['X']], data_transformed['y_log'])
y_pred_log = model_transformed.predict(data_transformed[['X']])
y_pred_original_scale = np.expm1(y_pred_log)
results['Transformation (Log)'] = {
    'coef': model_transformed.coef_[0],
    'intercept': model_transformed.intercept_,
    'r2': r2_score(data['y'], y_pred_original_scale),
    'rmse': np.sqrt(mean_squared_error(data['y'], y_pred_original_scale))
}

# --- Method 4: Imputation ---
# Replace the outlier values with a more representative value, like the median of the non-outlier data.
data_imputed = data.copy()
median_y = data_removed['y'].median() # Median of data *after* removing outliers
data_imputed.loc[outlier_indices, 'y'] = median_y
model_imputed = LinearRegression().fit(data_imputed[['X']], data_imputed['y'])
results['Imputation (Median)'] = {
    'coef': model_imputed.coef_[0],
    'intercept': model_imputed.intercept_,
    'r2': r2_score(data_imputed['y'], model_imputed.predict(data_imputed[['X']])),
    'rmse': np.sqrt(mean_squared_error(data_imputed['y'], model_imputed.predict(data_imputed[['X']])))
}

# --- Method 5: Use a Robust Model ---
# HuberRegressor is less sensitive to outliers. It applies a linear loss to samples classified as outliers.
model_robust = HuberRegressor().fit(data[['X']], data['y'])
y_pred_robust = model_robust.predict(data[['X']])
results['Robust Model (Huber)'] = {
    'coef': model_robust.coef_[0],
    'intercept': model_robust.intercept_,
    'r2': r2_score(data['y'], y_pred_robust),
    'rmse': np.sqrt(mean_squared_error(data['y'], y_pred_robust))
}


# --- 4. FINAL VISUALIZATION AND COMPARISON ---

# Create a DataFrame from the results for easy comparison
results_df = pd.DataFrame(results).T
print("--- Comparison of Model Performance ---")
print(results_df.round(4))

# Visualize the final fits
plt.figure(figsize=(14, 8))
# Plot original data
plt.scatter(data['X'], data['y'], color='gray', alpha=0.5, label='Original Data')
plt.scatter(data.loc[outlier_indices]['X'], data.loc[outlier_indices]['y'], color='red', s=100, label='Detected Outliers')

# Plot the regression lines
X_plot = np.linspace(0, 10, 100).reshape(-1, 1)

plt.plot(X_plot, model_baseline.predict(X_plot), 'r--', label=f"Baseline (RMSE: {results['Baseline']['rmse']:.2f})")
plt.plot(X_plot, model_removed.predict(X_plot), 'g-', linewidth=2, label=f"Removal (RMSE: {results['Removal']['rmse']:.2f})")
plt.plot(X_plot, model_robust.predict(X_plot), 'purple', linewidth=2, label=f"Robust Huber (RMSE: {results['Robust Model (Huber)']['rmse']:.2f})")

y_plot_transformed = np.expm1(model_transformed.predict(X_plot))
plt.plot(X_plot, y_plot_transformed, 'orange', linestyle=':', linewidth=2, label=f"Transformation (RMSE: {results['Transformation (Log)']['rmse']:.2f})")

plt.title('Comparison of Outlier Handling Methods')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.ylim(-5, 45) # Set y-axis limits for better visualization
plt.show()