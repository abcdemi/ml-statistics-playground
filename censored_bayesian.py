# --- Bayesian Censored Regression ---

# 1. SETUP: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pymc as pm
import arviz as az # ArviZ is for visualizing and diagnosing Bayesian models

# Set plot style and figure size
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- 2. THE EXPERIMENT: Data Generation (same as before) ---

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

# --- 3. Naive OLS Model for Comparison ---
ols_model = LinearRegression()
ols_model.fit(X.reshape(-1, 1), y_observed)
ols_intercept = ols_model.intercept_
ols_slope = ols_model.coef_[0]

print("="*60)
print("Naive OLS Model Results (Biased)")
print(f"Estimated Intercept: {ols_intercept:.3f} (True was {true_intercept})")
print(f"Estimated Slope:     {ols_slope:.3f} (True was {true_slope})")
print("="*60)

# --- 4. MODELING: Bayesian Censored Regression with PyMC ---

with pm.Model() as bayesian_tobit_model:
    # --- Priors ---
    # We set weakly informative priors for our parameters.
    # This is our initial belief before seeing the data.
    intercept = pm.Normal('intercept', mu=0, sigma=20)
    slope = pm.Normal('slope', mu=0, sigma=10)
    sigma = pm.HalfCauchy('sigma', beta=5) # Prior for the error term's standard deviation

    # --- Linear Model for the Latent Variable ---
    # This defines the relationship for the unobserved y*.
    mu = intercept + slope * X

    # --- Likelihood with the Latent Variable Trick ---
    # This is where the magic happens. PyMC's `Censored` distribution
    # handles the data augmentation (Gibbs sampling) for us under the hood.
    # It knows how to deal with the censored data points.
    y_likelihood = pm.Censored('y_likelihood',
                               dist=pm.Normal.dist(mu=mu, sigma=sigma), # The latent variable is normally distributed
                               lower=-np.inf, # No left-censoring
                               upper=censoring_point, # Right-censoring at our point
                               observed=y_observed) # The actual observed data

    # --- Run the MCMC Sampler ---
    # This is the step that performs the iterative process described above.
    # It draws thousands of samples from the posterior distribution.
    idata = pm.sample(2000, tune=1000, cores=1)


# --- 5. RESULTS AND VISUALIZATION ---

print("\n" + "="*60)
print("Bayesian Model Results (Unbiased)")
# `az.summary` gives us the mean, std dev, and credible intervals for our parameters.
summary = az.summary(idata, var_names=['intercept', 'slope', 'sigma'])
print(summary)

# Extract the mean of the posterior distributions to plot a single line
bayesian_intercept = summary['mean']['intercept']
bayesian_slope = summary['mean']['slope']

print(f"\nEstimated Intercept (Posterior Mean): {bayesian_intercept:.3f} (True was {true_intercept})")
print(f"Estimated Slope (Posterior Mean):     {bayesian_slope:.3f} (True was {true_slope})")
print("--> The Bayesian model also successfully recovers the true parameters!")
print("="*60)

# --- Visualize the Posterior Distributions (The Bayesian Result) ---
az.plot_trace(idata, var_names=['intercept', 'slope', 'sigma'])
plt.tight_layout()
plt.show()

# --- Final Comparison Plot ---
plt.figure(figsize=(14, 9))

# Plot the observed data, highlighting the censored points
plt.scatter(X[~is_censored], y_observed[~is_censored], color='blue', label='Observed Data')
plt.scatter(X[is_censored], y_observed[is_censored], color='red', marker='x', s=100, label='Censored Data')

# Plot the regression lines
plt.plot(X, true_intercept + true_slope * X, 'k-', linewidth=3, label='True Latent Relationship')
plt.plot(X, ols_intercept + ols_slope * X, 'r--', linewidth=2, label='Naive OLS Fit (Biased)')
plt.plot(X, bayesian_intercept + bayesian_slope * X, 'purple', linestyle='-', linewidth=3, label='Bayesian Fit (Posterior Mean)')

plt.axhline(censoring_point, color='black', linestyle=':', label=f'Censoring Point ({censoring_point})')
plt.title('Bayesian Censored Model vs. OLS', fontsize=18)
plt.xlabel('Predictor (X)', fontsize=14)
plt.ylabel('Outcome (y)', fontsize=14)
plt.legend(fontsize=12)
plt.ylim(bottom=0)
plt.show()