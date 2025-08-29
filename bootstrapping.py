# --- Bootstrapping ---

# 1. SETUP: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Set plot style and figure size
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==============================================================================
# --- PART 1: Estimate Standard Error & Confidence Interval for the Median ---
# ==============================================================================
print("="*60)
print("Part 1: Estimating Standard Error & CI for the Median")
print("="*60)

# --- The Problem: A small, non-normal dataset ---
# The formula for the standard error of the mean is simple (s/sqrt(n)), but for the median, it's complex and relies on assumptions.
# Bootstrapping is the perfect tool here.
np.random.seed(42)
# Create skewed data (log-normal distribution)
data = np.random.lognormal(mean=2, sigma=0.8, size=30)

# Calculate the observed median from our original sample
observed_median = np.median(data)
print(f"Original Sample Median: {observed_median:.2f}")

# --- The Bootstrap Process ---
n_bootstraps = 10000
bootstrap_medians = []

for i in range(n_bootstraps):
    # 1. Create a bootstrap sample: resample the original data WITH replacement
    bootstrap_sample = resample(data)
    # 2. Calculate the statistic of interest (median) for the bootstrap sample
    median = np.median(bootstrap_sample)
    # 3. Store the result
    bootstrap_medians.append(median)

# --- Analyze the Bootstrap Distribution ---
# The distribution of our bootstrap_medians is an estimate of the sampling distribution of the median.

# 1. Estimate the Standard Error
# The standard error of our statistic is simply the standard deviation of the bootstrap distribution.
bootstrap_se = np.std(bootstrap_medians)
print(f"Bootstrapped Standard Error of the Median: {bootstrap_se:.2f}")

# 2. Construct a Confidence Interval
# A 95% confidence interval is found by taking the 2.5th and 97.5th percentiles of the bootstrap distribution.
confidence_interval = np.percentile(bootstrap_medians, [2.5, 97.5])
print(f"95% Confidence Interval for the Median: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")

# --- Visualize the results ---
plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_medians, kde=True)
plt.axvline(observed_median, color='red', linestyle='-', linewidth=2, label=f'Observed Median ({observed_median:.2f})')
plt.axvline(confidence_interval[0], color='black', linestyle='--', label='95% CI Lower Bound')
plt.axvline(confidence_interval[1], color='black', linestyle='--', label='95% CI Upper Bound')
plt.title('Bootstrap Distribution of the Median')
plt.xlabel('Median Value')
plt.legend()
plt.show()


# ==============================================================================
# --- PART 2: Bootstrap Hypothesis Testing ---
# ==============================================================================
print("\n" + "="*60)
print("Part 2: Bootstrap Hypothesis Testing (Comparing Medians)")
print("="*60)

# --- The Problem: Compare two groups, A and B. Is the median of B significantly higher than A? ---
# A standard t-test assumes normality, which our data violates.
group_A = np.random.lognormal(mean=2.0, sigma=0.8, size=25)
group_B = np.random.lognormal(mean=2.3, sigma=0.8, size=30) # Group B has a slightly higher mean in generation

# Null Hypothesis (H0): The medians of the populations from which A and B are drawn are identical.
# The observed difference in medians is just due to random chance.
observed_diff = np.median(group_B) - np.median(group_A)
print(f"Observed difference in medians (B - A): {observed_diff:.2f}")

# --- The Bootstrap Process for Hypothesis Testing ---
# 1. Combine the groups to simulate the null hypothesis (that they come from the same population).
combined_data = np.concatenate([group_A, group_B])

# 2. Generate many bootstrap replicates under the null hypothesis.
n_replicates = 10000
bootstrap_diffs = []

for i in range(n_replicates):
    # Resample from the COMBINED data to create new simulated groups A and B
    simulated_A = np.random.choice(combined_data, size=len(group_A), replace=True)
    simulated_B = np.random.choice(combined_data, size=len(group_B), replace=True)
    # Calculate the difference in medians for this simulated reality
    diff = np.median(simulated_B) - np.median(simulated_A)
    bootstrap_diffs.append(diff)

# 3. Calculate the p-value
# The p-value is the fraction of simulated differences that are as extreme or more extreme than the observed difference.
p_value = np.sum(np.array(bootstrap_diffs) >= observed_diff) / n_replicates
print(f"Calculated p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: The p-value is less than 0.05. We reject the null hypothesis.")
else:
    print("Result: The p-value is not less than 0.05. We fail to reject the null hypothesis.")

# --- Visualize the results ---
plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_diffs, kde=True)
plt.axvline(observed_diff, color='red', linestyle='-', linewidth=2, label=f'Observed Difference ({observed_diff:.2f})')
plt.title('Bootstrap Distribution of Difference in Medians (Under Null Hypothesis)')
plt.xlabel('Difference (Median B - Median A)')
plt.legend()
plt.show()

# ==============================================================================
# --- PART 3: Confidence Interval for a Complex Model's Performance ---
# ==============================================================================
print("\n" + "="*60)
print("Part 3: CI for a Random Forest's R-squared Score")
print("="*60)

# --- The Problem: We have a Random Forest model. What is a reliable estimate of its R² performance on new data? ---
# There is no simple analytical formula for the confidence interval of R² for a Random Forest.

# Generate some non-linear data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.3, 100)

# We must have a held-out test set that is NEVER resampled.
# We bootstrap the TRAINING set to simulate different training realities.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- The Bootstrap Process ---
n_iterations = 500 # Fewer iterations due to model training time
bootstrap_r2_scores = []
train_indices = np.arange(len(X_train))

for i in range(n_iterations):
    # 1. Create a bootstrap sample of the TRAINING data indices
    bootstrap_indices = resample(train_indices)
    X_train_boot = X_train[bootstrap_indices]
    y_train_boot = y_train[bootstrap_indices]

    # 2. Train the complex model on the bootstrap sample
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_train_boot, y_train_boot)

    # 3. Evaluate on the UNTOUCHED test set and store the R² score
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    bootstrap_r2_scores.append(r2)

# Calculate the confidence interval for the R² score
r2_confidence_interval = np.percentile(bootstrap_r2_scores, [2.5, 97.5])

# --- Visualize the results ---
plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_r2_scores, kde=True)
plt.axvline(r2_confidence_interval[0], color='black', linestyle='--', label=f'95% CI Lower ({r2_confidence_interval[0]:.3f})')
plt.axvline(r2_confidence_interval[1], color='black', linestyle='--', label=f'95% CI Upper ({r2_confidence_interval[1]:.3f})')
plt.title('Bootstrap Distribution of Random Forest R-squared Scores')
plt.xlabel('R-squared on Test Set')
plt.legend()
plt.show()

print(f"Original R² on test set (from one model): {r2_score(y_test, RandomForestRegressor(random_state=42).fit(X_train, y_train).predict(X_test)):.3f}")
print(f"Bootstrapped 95% Confidence Interval for R²: [{r2_confidence_interval[0]:.3f}, {r2_confidence_interval[1]:.3f}]")
print("\nThis gives us a much more reliable estimate of how our model is likely to perform on new data.")