# ml_statistics_playground.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import scipy.stats as stats

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# --- UTILITY FUNCTION FOR MODEL EVALUATION ---
def evaluate_model(y_true, y_pred, model_name="Model"):
    """Prints and returns key regression metrics."""
    r2 = metrics.r2_score(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    
    print(f"--- {model_name} Evaluation ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    
    return {"R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae}

# --- UTILITY FUNCTION FOR RESIDUAL ANALYSIS ---
def perform_residual_analysis(y_true, y_pred, model_name="Model"):
    """Performs and plots residual analysis."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Residuals vs. Fitted Plot
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0])
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs. Fitted Values")
    
    # 2. Histogram of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_xlabel("Residuals")
    axes[1].set_title("Histogram of Residuals")

    # 3. Q-Q Plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title("Normal Q-Q Plot")
    
    fig.suptitle(f"Residual Analysis for {model_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- EXPERIMENT 1: LINEAR DATA & CLOSED-FORM SOLUTION ---
print("="*50)
print("EXPERIMENT 1: Linear Data with Closed-Form Solution")
print("="*50)

# 1.1 Generate Synthetic Linear Data
np.random.seed(42)
X_linear = 2 * np.random.rand(100, 1)
y_linear = 4 + 3 * X_linear + np.random.randn(100, 1) # y = 4 + 3x + noise

# Split data
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

# 1.2 Closed-Form Solution (Normal Equation)
# Add x0 = 1 to each instance
X_b = np.c_[np.ones((len(X_train_l), 1)), X_train_l] 
# Calculate theta using the Normal Equation: (X^T * X)^-1 * X^T * y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train_l)

print(f"Closed-Form Solution (Normal Equation) Coefficients:")
print(f"Intercept (β₀): {theta_best[0][0]:.4f}, Slope (β₁): {theta_best[1][0]:.4f}\n")

# 1.3 Model Building with Scikit-learn (which uses a similar closed-form approach)
lin_reg = LinearRegression()
lin_reg.fit(X_train_l, y_train_l)

print(f"Scikit-learn LinearRegression Coefficients:")
print(f"Intercept (β₀): {lin_reg.intercept_[0]:.4f}, Slope (β₁): {lin_reg.coef_[0][0]:.4f}\n")

# 1.4 Model Evaluation
y_pred_l = lin_reg.predict(X_test_l)
evaluate_model(y_test_l, y_pred_l, model_name="Linear Regression")

# 1.5 Data and Model Visualization
plt.figure(figsize=(12, 7))
plt.scatter(X_linear, y_linear, alpha=0.6, label="Data Points")
plt.plot(X_test_l, y_pred_l, color='red', linewidth=3, label="Regression Line")
plt.title("Linear Regression Fit")
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (y)")
plt.legend()
plt.show()

# 1.6 Residual Analysis
y_train_pred_l = lin_reg.predict(X_train_l)
perform_residual_analysis(y_train_l.flatten(), y_train_pred_l.flatten(), model_name="Linear Regression")

# 1.7 Cross-Validation
# Use negative mean squared error as the scoring metric because scikit-learn's cross_val_score expects a utility function (higher is better)
cv_scores = cross_val_score(lin_reg, X_linear, y_linear, cv=10, scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)

print("--- Cross-Validation (10-fold) ---")
print(f"RMSE Scores for each fold: {cv_rmse_scores}")
print(f"Mean RMSE: {cv_rmse_scores.mean():.4f}")
print(f"Standard Deviation of RMSE: {cv_rmse_scores.std():.4f}\n")

# --- EXPERIMENT 2: NON-LINEAR DATA & CASES WITHOUT A SIMPLE CLOSED-FORM SOLUTION ---
print("\n" + "="*50)
print("EXPERIMENT 2: Non-Linear Data")
print("="*50)

# 2.1 Generate Synthetic Non-Linear Data
np.random.seed(42)
m = 100
X_nl = 6 * np.random.rand(m, 1) - 3
y_nl = 0.5 * X_nl**2 + X_nl + 2 + np.random.randn(m, 1) # y = 0.5x^2 + x + 2 + noise

# Split data
X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(X_nl, y_nl, test_size=0.2, random_state=42)

# 2.2 Attempt a simple linear model (poor fit)
lin_reg_nl = LinearRegression()
lin_reg_nl.fit(X_train_nl, y_train_nl)
y_pred_nl_linear = lin_reg_nl.predict(X_test_nl)

print("--- Evaluation of Simple Linear Model on Non-Linear Data ---")
evaluate_model(y_test_nl, y_pred_nl_linear, "Simple Linear Regression")

# 2.3 Build a Polynomial Regression model
# This transforms features (e.g., x -> x, x^2), then applies a linear model.
# This is how we can model non-linear relationships without abandoning linear models entirely.
poly_reg = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
poly_reg.fit(X_train_nl, y_train_nl)

# 2.4 Evaluate the Polynomial Model
y_pred_nl_poly = poly_reg.predict(X_test_nl)
evaluate_model(y_test_nl, y_pred_nl_poly, "Polynomial Regression (degree=2)")

# 2.5 Data and Model Visualization
# Sort values for a smooth curve plot
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
y_plot_linear = lin_reg_nl.predict(X_plot)
y_plot_poly = poly_reg.predict(X_plot)

plt.figure(figsize=(12, 7))
plt.scatter(X_nl, y_nl, alpha=0.6, label="Data Points")
plt.plot(X_plot, y_plot_linear, color='red', linewidth=2, linestyle="--", label="Poor Linear Fit")
plt.plot(X_plot, y_plot_poly, color='green', linewidth=3, label="Good Polynomial Fit")
plt.title("Linear vs. Polynomial Regression on Non-Linear Data")
plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (y)")
plt.legend()
plt.show()

# 2.6 Residual Analysis for Polynomial Model
y_train_pred_nl = poly_reg.predict(X_train_nl)
perform_residual_analysis(y_train_nl.flatten(), y_train_pred_nl.flatten(), model_name="Polynomial Regression")