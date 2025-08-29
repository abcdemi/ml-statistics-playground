# --- Tobit Model and Censored Data Demonstration (Corrected Final) ---

# 1. SETUP: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.base.model import GenericLikelihoodModel # Using GenericLikelihoodModel for .fit()

# Set plot style and figure size
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- 2. THE EXPERIMENT: Data Generation ---

# Step 1: Define the TRUE underlying linear relationship for the latent variable.
np.random.seed(42)
n_samples = 200
true_intercept = 5
true_slope = 2
X = np.linspace(0, 10, n_samples)
error = np.random.normal(0, 3, n_samples)

# Step 2: Create the latent (unobserved) variable y*
y_latent = true_intercept + true_slope * X + error

# Step 3: Create the OBSERVED variable y by censoring the latent variable.
censoring_point = 20
y_observed = np.copy(y_latent)
y_observed[y_observed > censoring_point] = censoring_point
is_censored = (y_latent > censoring_point)

# --- 3. MODELING: Comparing Naive OLS vs. Tobit ---

# --- Model 1: The Naive OLS Regression ---
ols_model = LinearRegression()
ols_model.fit(X.reshape(-1, 1), y_observed)
ols_intercept = ols_model.intercept_
ols_slope = ols_model.coef_[0]

print("="*60)
print("Naive OLS Model Results (Biased)")
print(f"Estimated Intercept: {ols_intercept:.3f} (True was {true_intercept})")
print(f"Estimated Slope:     {ols_slope:.3f} (True was {true_slope})")
print("--> OLS significantly underestimates the slope and overestimates the intercept.")
print("="*60)


# --- Model 2: The Tobit Model ---
# This class now correctly defines `loglikeobs` for GenericLikelihoodModel.
class Tobit(GenericLikelihoodModel):
    def __init__(self, endog, exog, censor_point, **kwds):
        super(Tobit, self).__init__(endog, exog, **kwds)
        self.censor_point = censor_point

    # Define loglikeobs for per-observation log-likelihood ---
    def loglikeobs(self, params):
        beta = params[:-1]
        sigma = np.exp(params[-1]) # Ensure sigma > 0

        # Numerical stability check for sigma: must be strictly positive
        if sigma <= 1e-8:
            return np.full(self.endog.shape, -np.inf) # Return array of -inf for invalid sigma

        mu = self.exog @ beta

        uncensored_idx = self.endog < self.censor_point
        censored_idx = ~uncensored_idx

        # Initialize array for individual log-likelihoods
        ll_individual = np.zeros_like(self.endog, dtype=float)

        # Log-likelihood for uncensored observations: Normal PDF
        ll_individual[uncensored_idx] = norm.logpdf(self.endog[uncensored_idx], loc=mu[uncensored_idx], scale=sigma)

        # Log-likelihood for censored observations: Log of Survival Function (P(Y* > c))
        ll_individual[censored_idx] = norm.logsf(self.censor_point, loc=mu[censored_idx], scale=sigma)

        return ll_individual

    # Define loglike as the sum of loglikeobs (often implicitly handled if loglikeobs exists, but good to be explicit)
    def loglike(self, params):
        return np.sum(self.loglikeobs(params))

    # The predict method on the model instance, for clarity.
    # statsmodels.regression.linear_model.RegressionResults.predict is usually preferred.
    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        beta = params[:-1]
        # For Tobit, we predict the latent variable value
        return exog @ beta

# Fit the Tobit model
X_const = sm.add_constant(X)
# Start parameters: [intercept, slope, log(sigma)]
start_params = [ols_intercept, ols_slope, np.log(np.std(y_observed))]
tobit_model_instance = Tobit(y_observed, X_const, censoring_point)
tobit_fit_results = tobit_model_instance.fit(start_params=start_params)
tobit_params = tobit_fit_results.params
tobit_intercept = tobit_params[0]
tobit_slope = tobit_params[1]

print("\n" + "="*60)
print("Tobit Model Results (Unbiased)")
print(f"Estimated Intercept: {tobit_intercept:.3f} (True was {true_intercept})")
print(f"Estimated Slope:     {tobit_slope:.3f} (True was {true_slope})")
print("--> The Tobit model successfully recovers the true underlying parameters!")
print("="*60)


# --- 4. VISUALIZATION: The Proof ---

plt.figure(figsize=(14, 9))

# Plot the "true" latent data to show the real relationship
plt.scatter(X, y_latent, color='gray', alpha=0.3, label='Latent (Unobserved) Data')

# Plot the observed data, highlighting the censored points
plt.scatter(X[~is_censored], y_observed[~is_censored], color='blue', label='Observed Data')
plt.scatter(X[is_censored], y_observed[is_censored], color='red', marker='x', s=100, label='Censored Data')

# Plot the regression lines
plt.plot(X, true_intercept + true_slope * X, 'k-', linewidth=3, label='True Latent Relationship')
plt.plot(X, ols_intercept + ols_slope * X, 'r--', linewidth=2, label='Naive OLS Fit (Biased)')
plt.plot(X, tobit_intercept + tobit_slope * X, 'g-', linewidth=3, label='Tobit Fit (Corrected)')

plt.axhline(censoring_point, color='black', linestyle=':', label=f'Censoring Point ({censoring_point})')
plt.title('Tobit Model vs. OLS on Censored Data', fontsize=18)
plt.xlabel('Predictor (X)', fontsize=14)
plt.ylabel('Outcome (y)', fontsize=14)
plt.legend(fontsize=12)
plt.ylim(bottom=0)
plt.show()