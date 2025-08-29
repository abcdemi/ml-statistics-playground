# --- Advanced Statistics ---

# 1. SETUP: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Set plot style and figure size
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# ==============================================================================
# --- PART 1: Estimating Prior Parameters with Empirical Bayes ---
# ==============================================================================
print("="*60)
print("Part 1: Empirical Bayes for Success Rate Estimation")
print("="*60)

# --- The Problem Scenario ---
# We have data on 100 different online courses. For each, we know the number of students
# (n_trials) and how many passed the final exam (n_successes).
# Some courses have thousands of students, others have only a few.
# Goal: Find the "true" pass rate for each course, avoiding extreme estimates for courses with little data.

np.random.seed(42)
n_courses = 100
# Each course has a true, underlying pass rate (p), which we assume comes from a Beta distribution.
# This Beta distribution is our "prior".
true_alpha, true_beta = 5, 20 # Parameters of the true prior (unknown to us)
true_pass_rates = np.random.beta(true_alpha, true_beta, n_courses)

# The number of students per course is highly variable.
n_trials = np.random.randint(5, 5000, n_courses)
# The observed number of successes is a draw from a Binomial distribution.
n_successes = np.random.binomial(n_trials, true_pass_rates)

# Create a DataFrame
courses = pd.DataFrame({
    'course_id': range(n_courses),
    'n_trials': n_trials,
    'n_successes': n_successes
})

# --- Approach 1: Naive Estimate (Maximum Likelihood Estimate) ---
# This is simply the observed success rate. It's unreliable for small n_trials.
courses['naive_rate'] = courses['n_successes'] / courses['n_trials']

# --- Approach 2: Empirical Bayes with a Beta-Binomial Model ---
# We use the data itself to estimate the parameters (alpha_0, beta_0) of the Beta prior.

# Step 1: Estimate prior parameters using the Method of Moments.
# We calculate the mean and variance of the naive rates to estimate alpha_0 and beta_0.
mu = courses['naive_rate'].mean()
var = courses['naive_rate'].var()

# Formulas to solve for alpha and beta from mean (mu) and variance (var)
common_term = (mu * (1 - mu) / var) - 1
alpha_0 = mu * common_term
beta_0 = (1 - mu) * common_term

print(f"Empirically Estimated Prior Parameters:")
print(f"Alpha_0: {alpha_0:.2f} (True was {true_alpha})")
print(f"Beta_0: {beta_0:.2f} (True was {true_beta})\n")

# Step 2: Apply the Bayesian update to get the "shrunken" posterior estimate for each course.
# For a Beta-Binomial model, the posterior mean is (alpha_0 + k) / (alpha_0 + beta_0 + n).
courses['eb_rate'] = (alpha_0 + courses['n_successes']) / (alpha_0 + beta_0 + courses['n_trials'])

# --- Visualization ---
plt.figure(figsize=(14, 8))
# Use size to represent the number of trials
sns.scatterplot(x='naive_rate', y='eb_rate', data=courses, size='n_trials',
                sizes=(20, 1000), alpha=0.7, legend='brief')

# Add a line for reference (y=x)
plt.plot([0, 1], [0, 1], 'r--', label='y=x (No Shrinkage)')
plt.axhline(mu, color='gray', linestyle=':', label=f'Overall Mean ({mu:.2f})')

plt.title('Empirical Bayes Shrinkage of Course Pass Rates', fontsize=16)
plt.xlabel('Naive Estimate (Successes / Trials)')
plt.ylabel('Empirical Bayes Estimate (Shrunken)')
plt.legend()
plt.xlim(-0.05, 0.5)
plt.ylim(-0.05, 0.5)
plt.show()

print("Notice how courses with few trials (small dots) are 'shrunken' heavily towards the overall mean,")
print("while courses with many trials (large dots) stay close to their naive estimate.")


# ==============================================================================
# --- PART 2: Link Functions in Generalized Linear Models (GLMs) ---
# ==============================================================================
print("\n" + "="*60)
print("Part 2: Demonstrating GLM Link Functions")
print("="*60)

# --- Scenario A: Predicting a Probability (Logistic Regression) ---
# The target is binary (0 or 1). The mean is a probability (bounded between 0 and 1).
# We need a LINK function to map the linear model's unbounded output to this [0, 1] range.
# Family: Binomial, Link: Logit (log(p/(1-p)))

np.random.seed(42)
X1 = np.linspace(-5, 5, 100)
# The linear part
linear_predictor = 0.8 * X1 + 0.5
# Pass through the inverse of the logit link (the logistic function) to get probabilities
probs = 1 / (1 + np.exp(-linear_predictor))
# Generate binary outcomes
y1 = np.random.binomial(1, probs)

# Fit a GLM
X1_const = sm.add_constant(X1)
glm_binom = sm.GLM(y1, X1_const, family=sm.families.Binomial()) # Link defaults to logit
res_binom = glm_binom.fit()
y_pred_probs = res_binom.predict(X1_const)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X1, y1, alpha=0.5, label='Observed Data (0 or 1)')
plt.plot(X1, y_pred_probs, color='red', label='GLM Prediction (Probability)')
plt.title('GLM with Logit Link for Binary Data (Logistic Regression)')
plt.xlabel('Predictor (e.g., Study Hours)')
plt.ylabel('Outcome (Probability of Passing)')
plt.legend()
plt.show()
print("The LOGIT link function maps the linear model's output to the S-shaped curve, ensuring predictions are valid probabilities [0, 1].")


# --- Scenario B: Predicting Counts (Poisson Regression) ---
# The target is a non-negative integer (0, 1, 2, ...). The mean is a rate (bounded at >= 0).
# We need a LINK function to map the linear model's unbounded output to this [0, inf) range.
# Family: Poisson, Link: Log (log(rate))

np.random.seed(42)
X2 = np.linspace(0, 4, 100)
# The linear part
linear_predictor2 = 0.6 * X2 + 0.2
# Pass through the inverse of the log link (the exponential function) to get rates
rates = np.exp(linear_predictor2)
# Generate count outcomes
y2 = np.random.poisson(rates)

# Fit a GLM
X2_const = sm.add_constant(X2)
glm_poisson = sm.GLM(y2, X2_const, family=sm.families.Poisson()) # Link defaults to log
res_poisson = glm_poisson.fit()
y_pred_rates = res_poisson.predict(X2_const)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X2, y2, alpha=0.5, label='Observed Data (Counts)')
plt.plot(X2, y_pred_rates, color='red', label='GLM Prediction (Rate)')
plt.title('GLM with Log Link for Count Data (Poisson Regression)')
plt.xlabel('Predictor (e.g., Website Traffic)')
plt.ylabel('Outcome (Number of Signups)')
plt.legend()
plt.show()
print("The LOG link function maps the linear model's output to the exponential curve, ensuring predictions are valid rates (>= 0).")