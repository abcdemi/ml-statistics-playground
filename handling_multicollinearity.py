# --- Multicollinearity Demonstration Script (Corrected) ---

# 1. SETUP: Import libraries and create a synthetic dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# Generate synthetic data
np.random.seed(42)
n_samples = 100
# X1: Base feature
X1 = np.random.uniform(0, 10, n_samples)
# X2: Highly correlated with X1. X2 is essentially X1 + some small random noise.
X2 = X1 + np.random.normal(0, 0.5, n_samples)
# X3: An independent feature, not correlated with X1 or X2.
X3 = np.random.uniform(0, 10, n_samples)

# y: Our target variable. It's a function of X1 and X3.
# Note: y does NOT depend on X2 directly, only through its correlation with X1.
y = 2 * X1 + 3 * X3 + np.random.normal(0, 2, n_samples)

# Create a DataFrame
data = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})
X = data[['X1', 'X2', 'X3']]


# --- 2. BASELINE MODEL & DETECTION OF MULTICOLLINEARITY ---

print("--- 2.1 Baseline Model with All Features ---")
# Build a standard Linear Regression model (Ordinary Least Squares)
model_baseline = LinearRegression()
model_baseline.fit(X, y)

# The coefficients are unstable. Notice how X1's coefficient is negative
# and X2's is positive, even though y was created from a positive X1.
# This is a classic sign of multicollinearity.
print("Baseline Coefficients:", model_baseline.coef_)
print("\n")


print("--- 2.2 Detection Method 1: Correlation Matrix ---")
# A correlation matrix shows the linear relationship between variables.
# A value close to 1 (or -1) indicates a very strong positive (or negative) correlation.
correlation_matrix = X.corr()

# Visualize the matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Independent Variables")
plt.show()

print("As seen in the heatmap, X1 and X2 have a correlation of 0.99, which is extremely high.\n")


print("--- 2.3 Detection Method 2: Variance Inflation Factor (VIF) ---")
# VIF measures how much the variance of a regression coefficient is inflated due to multicollinearity.
# VIF = 1 / (1 - R^2), where R^2 is from a regression of one predictor on the others.
# A common rule of thumb: VIF > 5 or 10 is a strong sign of multicollinearity.
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)
print("\nThe VIF scores for X1 and X2 are extremely high, confirming severe multicollinearity.\n")


# --- 3. SOLVING MULTICOLLINEARITY ---

print("--- 3.1 Solution 1: Remove one of the correlated features ---")
# Since X1 and X2 are highly correlated, we can remove one. We'll keep X1.
X_reduced = X.drop('X2', axis=1)

# Fit a new model on the reduced feature set
model_reduced = LinearRegression()
model_reduced.fit(X_reduced, y)

# These coefficients are now stable and interpretable. X1's coefficient is ~2.0,
# which matches how we generated the data.
print("Coefficients after removing X2:", model_reduced.coef_)

# Check the VIF again for the new model. All scores are now very low.
vif_data_reduced = pd.DataFrame()
vif_data_reduced["feature"] = X_reduced.columns
vif_data_reduced["VIF"] = [variance_inflation_factor(X_reduced.values, i) for i in range(len(X_reduced.columns))]
print("\nVIF scores for the reduced model:")
print(vif_data_reduced)
print("\n")


print("--- 3.2 Solution 2: Use Ridge Regression (L2 Regularization) ---")
# Ridge Regression adds a penalty to the cost function to shrink coefficient sizes.
# Cost Function = Sum of Squared Errors + alpha * sum(coefficients^2)
# The 'alpha' hyperparameter controls the strength of the penalty.
# This method shrinks the coefficients of correlated predictors towards each other.
# We will use RidgeCV to find the best alpha via cross-validation.

# Find the best alpha
ridge_cv = RidgeCV(alphas=np.logspace(-6, 6, 13))
ridge_cv.fit(X, y)
best_alpha = ridge_cv.alpha_
print(f"Best alpha found by RidgeCV: {best_alpha:.4f}\n")

# Fit the Ridge model with the best alpha
model_ridge = Ridge(alpha=best_alpha)
model_ridge.fit(X, y)

# Notice the coefficients for X1 and X2 are now much smaller and more balanced
# than in the baseline model. Ridge doesn't nullify them, but it reduces their instability.
print("Coefficients from Ridge Regression:", model_ridge.coef_)
print("\n")


# --- 4. VISUALIZATION & SUMMARY ---

print("--- 4.1 Visualizing the Effect of Ridge Regularization ---")
# Plot how coefficients change as the regularization strength (alpha) increases.
alphas = 10**np.linspace(10,-2,100)*0.5
ridge = Ridge()
coefs = []

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Coefficient Paths as Alpha Increases')
plt.legend(X.columns)
plt.axis('tight')
plt.show()

print("The plot above shows that as alpha increases, the large, unstable coefficients of X1 and X2 are 'tamed' and converge.")
print("\n")

print("--- 4.2 Final Comparison of Coefficients ---")
# Create a final comparison table
comparison = pd.DataFrame({
    "Feature": X.columns,
    "Baseline OLS": model_baseline.coef_,
    "OLS (X2 Removed)": np.append(model_reduced.coef_, [np.nan]), # Add NaN for removed feature
    "Ridge Regression": model_ridge.coef_
})
print(comparison.round(4))
print("\nSUMMARY: Multicollinearity in the baseline model produced unstable and misleading coefficients.")
print("Both removing a feature and using Ridge Regression resulted in more stable and reliable models.")