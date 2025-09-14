# ML Statistics Playground

This repository is a collection of hands-on Python experiments and detailed explanations designed to build a deep, practical understanding of the core statistical concepts that underpin modern machine learning.

Each section covers a fundamental challenge in data modeling, explains the theory, and provides a self-contained Python script to demonstrate the problem and its solutions.

## Table of Contents

1.  [**Core Regression Concepts**](#1-core-regression-concepts)
    *   Linear Models: Assumptions & Solutions
    *   Model Evaluation, Residual Analysis & Cross-Validation
2.  [**Handling Common Data Challenges**](#2-handling-common-data-challenges)
    *   [Multicollinearity](#21-handling-multicollinearity)
    *   [Outliers](#22-handling-outliers)
    *   [Noise and Overfitting](#23-handling-noise-and-overfitting)
    *   [Skewed Datasets](#24-handling-skewed-datasets)
3.  [**Advanced Regression Techniques**](#3-advanced-regression-techniques)
    *   [Quantile Regression](#31-quantile-regression-modeling-the-full-picture)
    *   [Censored Data & Tobit Models](#32-censored-data--tobit-models)
4.  [**Model Validation & Inference**](#4-model-validation--inference)
    *   [Time Series Diagnostics](#41-time-series-diagnostics-ljung-box--newey-west)
    *   [Bootstrapping](#42-bootstrapping-inference-without-assumptions)

## 1. Core Regression Concepts

This section covers the foundational theory of linear regression models.

*   **Linear Models**: An overview of Simple and Multiple Linear Regression, including the core assumptions (Linearity, Independence, Homoscedasticity, Normality, No Multicollinearity).
*   **Model Fitting**:
    *   **Closed-Form Solution (Normal Equation)**: A direct, analytical solution for finding the optimal coefficients. Computationally efficient for smaller datasets.
    *   **Optimization (Gradient Descent)**: An iterative approach for minimizing the cost function, essential for large datasets and more complex models.
*   **Model Evaluation**:
    *   **Metrics**: R-squared (R²), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).
    *   **Residual Analysis**: The critical step of plotting residuals to diagnose violated model assumptions.
    *   **Cross-Validation**: A robust technique for estimating a model's performance on unseen data and preventing overfitting.

---

## 2. Handling Common Data Challenges

This section provides practical Python experiments for diagnosing and solving common data problems.

### 2.1 Handling Multicollinearity

*   **Problem**: When two or more predictor variables are highly correlated, leading to unstable and uninterpretable model coefficients.
*   **Detection**: Correlation Matrices and Variance Inflation Factor (VIF).
*   **Solutions**:
    1.  Removing one of the correlated features.
    2.  Using Regularization (e.g., Ridge Regression) to shrink coefficients.
*   **Script**: `handling_multicollinearity.py`
*   **Key Result**: The experiment shows how a baseline model produces nonsensical coefficients, which are corrected by both removal and regularization.

    | Correlation Matrix | Ridge Coefficient Paths |
    | :---: | :---: |
    | `[IMAGE_PLACEHOLDER_FOR_CORR_MATRIX.PNG]` | `[IMAGE_PLACEHOLDER_FOR_RIDGE_PATHS.PNG]` |

### 2.2 Handling Outliers

*   **Problem**: Outliers can dramatically skew the fit of a linear regression model, which minimizes *squared* errors.
*   **Detection**: Interquartile Range (IQR) on **model residuals**.
*   **Solutions**:
    1.  Correction (if it's a data entry error).
    2.  Removal (if the point is invalid).
    3.  Transformation (e.g., log transform).
    4.  Imputation (e.g., with the median).
    5.  Using Robust Models (e.g., Huber Regression).
*   **Script**: `handling_outliers.py`
*   **Key Result**: Demonstrates how a single outlier ruins an OLS fit and compares the effectiveness of different handling strategies, showing that Removal and Robust Models are often superior.

    | Outlier's Effect on OLS | Comparison of Solutions |
    | :---: | :---: |
    | `[IMAGE_PLACEHOLDER_FOR_OUTLIER_FIT.PNG]` | `[IMAGE_PLACEHOLDER_FOR_OUTLIER_SOLUTIONS.PNG]` |

### 2.3 Handling Noise and Overfitting

*   **Problem**: A model with high variance (like a deep Decision Tree) can learn the random noise in the training data, failing to generalize to new data.
*   **Solutions**:
    1.  **Regularization (Lasso)**: Performs feature selection by shrinking coefficients of noisy features to zero.
    2.  **Ensemble Methods (Random Forest, Gradient Boosting)**: Reduce variance by combining the predictions of many simpler models.
    3.  **Dimensionality Reduction (PCA)**: Filters noise by capturing the signal in a smaller number of components.
*   **Script**: `handling_noise.py`
*   **Key Result**: The plot vividly shows a Decision Tree overfitting to the noise, while Lasso, Random Forest, and Gradient Boosting successfully ignore the noise and learn the true underlying signal.

    | Comparison of Noise Handling Models |
    | :---: |
    | `[IMAGE_PLACEHOLDER_FOR_NOISE_HANDLING.PNG]` |

### 2.4 Handling Skewed Datasets

*   **Problem**: Linear models assume normally distributed residuals. Skewed data violates this, leading to poor performance.
*   **Detection**: Visual inspection (histogram) and calculating the skewness coefficient.
*   **Solutions**: Applying mathematical transformations to make the data more symmetric.
    *   **Right-Skew**: Log, Square Root, Box-Cox, Yeo-Johnson.
    *   **Left-Skew**: Square, Cube, Yeo-Johnson.
*   **Script**: `handling_skewed_data.py`
*   **Key Result**: Shows how transforming a skewed target variable can linearize the relationship, dramatically improving the performance of a Linear Regression model.

    | Original vs. Transformed Distributions |
    | :---: |
    | `[IMAGE_PLACEHOLDER_FOR_SKEW_TRANSFORMATIONS.PNG]` |

---

## 3. Advanced Regression Techniques

### 3.1 Quantile Regression: Modeling the Full Picture

*   **Concept**: While OLS models the conditional *mean*, Quantile Regression models the conditional *quantiles* (e.g., the median, 10th percentile, 90th percentile).
*   **Why it's useful**:
    1.  Robust to outliers (by modeling the median).
    2.  Perfect for understanding heteroscedasticity (when the data's spread changes).
    3.  Provides a full range of possible outcomes, not just the average.
*   **Script**: `quantile_regression.py`
*   **Key Result**: On heteroscedastic data, the plot shows OLS providing a single, uninformative line, while the quantile regression lines diverge, perfectly modeling the increasing variance and providing a reliable 90% prediction interval.

    | Quantile Regression vs. OLS |
    | :---: |
    | `[IMAGE_PLACEHOLDER_FOR_QUANTILE_REG.PNG]` |

### 3.2 Censored Data & Tobit Models

*   **Concept**: Censored data occurs when a value is only partially known (e.g., "5+ hours"). Naive models like OLS see "5+" as just "5" and produce severely biased results.
*   **The Latent Variable Trick**: The Tobit model assumes an unobserved, underlying linear relationship and uses a special likelihood function that accounts for the probability of an observation being censored.
*   **Bayesian Approach**: Bayesian inference naturally handles this by treating the censored values as missing data to be imputed in each MCMC step (data augmentation).
*   **Scripts**: `censored_tobit.py`, `censored_bayesian.py`
*   **Key Result**: The experiment proves that OLS fails completely, producing a flattened, biased line. Both the Frequentist (Tobit) and Bayesian models successfully ignore the censored pile-up and recover the true, unbiased underlying relationship.

    | Tobit & Bayesian vs. OLS |
    | :---: |
    | `[IMAGE_PLACEHOLDER_FOR_TOBIT_FIT.PNG]` |

---

## 4. Model Validation & Inference

### 4.1 Time Series Diagnostics: Ljung-Box & Newey-West

*   **Concept**: Time series models must be checked to ensure their residuals are free of autocorrelation (i.e., are white noise).
*   **ACF Plot**: Visualizes the "memory" of the series or residuals.
*   **Ljung-Box Test**: A formal statistical test for model adequacy. The null hypothesis is that the residuals are random. **A high p-value is good**, indicating the model is adequate.
*   **Newey-West (HAC) Estimators**: Corrects the standard errors of a model's coefficients to make them reliable even if autocorrelation is present.
*   **Script**: `time_series_diagnostics.py`
*   **Key Result**: The script fits a "bad" AR(1) model and a "good" AR(2) model, showing how the Ljung-Box test correctly rejects the bad model (low p-value) and accepts the good one (high p-value).

### 4.2 Bootstrapping: Inference Without Assumptions

*   **Concept**: A powerful simulation technique for estimating the uncertainty of any statistic (like a median, or even a model's R² score) without relying on traditional statistical formulas or assumptions.
*   **How it Works**: It repeatedly creates new "bootstrap samples" by sampling from the original data *with replacement*. The distribution of the statistic calculated on these thousands of samples approximates the true sampling distribution.
*   **Uses**:
    1.  Estimating Standard Errors.
    2.  Constructing Confidence Intervals.
    3.  Performing Hypothesis Tests.
*   **Script**: `bootstrapping.py`