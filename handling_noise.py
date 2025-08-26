# --- Noise Handling ---

# 1. SETUP: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Set plot style and figure size
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# --- 2. DATA GENERATION with NOISE ---

np.random.seed(42)
n_samples = 100

# Create the true, clean signal
X_true = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
y_true = np.sin(X_true).ravel()

# Add two types of noise:
# a) Noise in the target variable (measurement error)
y_noisy = y_true + np.random.normal(0, 0.5, y_true.shape)

# b) Irrelevant, noisy features
n_noise_features = 10
X_noise_features = np.random.rand(n_samples, n_noise_features)
# Combine the true feature with the noisy ones
X_full = np.hstack((X_true, X_noise_features))

# Split data into training and testing sets to evaluate generalization
X_train, X_test, y_train, y_test = train_test_split(X_full, y_noisy, test_size=0.3, random_state=42)

# --- 3. DEMONSTRATING THE PROBLEM: OVERFITTING TO NOISE ---

# A high-depth decision tree has high variance and will fit the noise in the training data
overfitting_model = DecisionTreeRegressor(max_depth=15)
overfitting_model.fit(X_train, y_train)

# --- 4. IMPLEMENTING SOLUTIONS TO HANDLE NOISE ---

# --- Solution 1: Regularization (Lasso) ---
# Lasso (L1 Regularization) adds a penalty equal to the absolute value of the magnitude of coefficients.
# Math: Cost = Sum of Squared Errors + alpha * sum(|coefficients|)
# This forces the coefficients of irrelevant (noisy) features towards zero, effectively selecting only the important ones.
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)

# --- Solution 2: Robust Ensemble Algorithm (Random Forest) ---
# Random Forest is an ensemble method using bagging.
# It builds many decision trees on different random subsets of the data and features.
# By averaging their predictions, it reduces variance and smooths out the noise.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# --- Solution 3: Dimensionality Reduction (PCA) ---
# PCA transforms the data to a new coordinate system of principal components.
# It captures the maximum variance (the signal) in the first few components,
# effectively filtering out the random, high-dimensional noise.
# We create a pipeline to first apply PCA and then fit a linear model.
pca_pipeline = Pipeline([
    ('pca', PCA(n_components=2)), # Reduce to 2 components from 11
    ('linear_regression', LinearRegression())
])
pca_pipeline.fit(X_train, y_train)


# --- 5. EVALUATION AND VISUALIZATION ---

# Store models and their test set predictions
models = {
    "Overfitting Decision Tree": overfitting_model,
    "Regularization (Lasso)": lasso_model,
    "Ensemble (Random Forest)": rf_model,
    "Dimensionality Reduction (PCA)": pca_pipeline
}

results = {}
# Sort test data for clean plotting
sort_idx = X_test[:, 0].argsort()
X_test_sorted = X_test[sort_idx]

plt.figure(figsize=(14, 8))
# Plot the ground truth and noisy data
plt.scatter(X_true, y_noisy, alpha=0.5, label='Noisy Data Points')
plt.plot(X_true, y_true, 'k-', linewidth=3, label='True Underlying Signal')

# Plot each model's predictions on the test set
for name, model in models.items():
    y_pred = model.predict(X_test_sorted)
    rmse = np.sqrt(mean_squared_error(y_test[sort_idx], y_pred))
    results[name] = rmse
    plt.plot(X_test_sorted[:, 0], y_pred, label=f"{name} (Test RMSE: {rmse:.3f})")

plt.title('Comparison of Models Handling Noisy Data')
plt.xlabel('True Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.ylim(-2, 2)
plt.show()

# --- Quantitative Comparison ---
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Test_RMSE']).sort_values('Test_RMSE')
print("--- Final Model Performance on Test Set ---")
print(results_df)

# Visualize the coefficients from Lasso to show feature selection
lasso_coefs = pd.DataFrame({
    'Feature': ['True Feature'] + [f'Noise Feature {i+1}' for i in range(n_noise_features)],
    'Coefficient': lasso_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n--- Lasso Coefficients (Feature Importance) ---")
print(lasso_coefs.round(4))
print("\nNote how Lasso correctly assigned a large coefficient to the true feature and shrunk the noisy ones to zero.")