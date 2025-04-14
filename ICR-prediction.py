import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.dummy import DummyRegressor
import warnings
from copy import deepcopy
import scipy.stats as stats

warnings.filterwarnings("ignore")  # Suppress warnings if needed

# =============================
# 1. Load Dataset
# =============================
file_path = "Dataset-A1.xlsx"  # Ensure correct file path
df_new = pd.read_excel(file_path)

# Define features and target
features = ['RQD', 'RMR', 'UCS']
target = 'ICR'

# Drop rows with missing values
df_cleaned = df_new.dropna(subset=features + [target])

# Extract features and target
X = df_cleaned[features]
y = df_cleaned[target]

# =============================
# 2. Normalize Data
# =============================
def normalize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

X_scaled, scaler = normalize_data(X)

# =============================
# 3. Generate Polynomial Features
# =============================
def generate_polynomial_features(X, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly, poly


X_poly, poly = generate_polynomial_features(X_scaled, degree=2)

# For ease of interpretation, attempt to get feature names
try:
    feature_names = poly.get_feature_names_out(features)
except AttributeError:
    feature_names = [f"Feature_{i}" for i in range(X_poly.shape[1])]

# -----------------------------
# Custom Titles for Features
# -----------------------------
custom_titles = {
    'RQD': 'RQD Distribution',
    'RMR': 'RMR value Distribution',
    'UCS': 'UCS Distribution',
}


# =============================
# 4. Synthetic Data Generation Methods
# =============================
def synthesize_data_ridge(X, y, n_samples=1000, noise_factor=0.2):
    bootstrapped_X = resample(X, replace=True, n_samples=n_samples, random_state=42)
    ridge_temp = Ridge(alpha=0.5)
    ridge_temp.fit(X, y)
    y_synthetic = ridge_temp.predict(bootstrapped_X)
    feature_std = np.std(X, axis=0)
    noise = noise_factor * np.random.randn(*bootstrapped_X.shape) * feature_std
    bootstrapped_X += noise
    return bootstrapped_X, y_synthetic


def synthesize_data_gaussian(X, y, n_samples=1000, noise_std=0.2):
    bootstrapped_X = resample(X, replace=True, n_samples=n_samples, random_state=42)
    y_synthetic = resample(y, replace=True, n_samples=n_samples, random_state=42)
    y_synthetic += np.random.normal(loc=0, scale=noise_std, size=len(y_synthetic))
    return bootstrapped_X, y_synthetic


def synthesize_data_rf(X, y, n_samples=1000, noise_factor=0.2):
    bootstrapped_X = resample(X, replace=True, n_samples=n_samples, random_state=42)
    # Use default RF hyperparameters for data synthesis (looser than tightened ones)
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_temp.fit(X, y)
    y_synthetic = rf_temp.predict(bootstrapped_X)
    bootstrapped_X += noise_factor * np.random.randn(*bootstrapped_X.shape)
    return bootstrapped_X, y_synthetic


synthetic_methods = {
    "Ridge Regression Labeling": synthesize_data_ridge,
    "Gaussian Noise Labeling": synthesize_data_gaussian,
    "Random Forest Regression Labeling": synthesize_data_rf
}

# =============================
# 5. Model Training & Performance Evaluation
# =============================
def evaluate_models(X_train, y_train, X_test, y_test):
    """
    Trains a collection of models on synthetic data and evaluates them on the original data.
    Returns a dictionary of performance metrics.
    """
    local_models = {
        'Ridge': Ridge(alpha=0.5),
        'Lasso': Lasso(alpha=0.01, max_iter=1000),
        'ElasticNet': ElasticNet(random_state=42),
        # Tightened RF model for prediction
        'Random Forest': RandomForestRegressor(
            n_estimators=50,  # increased
            max_depth=4,  # decreased
            min_samples_split=50,  # increased
            min_samples_leaf=10,  # increased
            max_features='sqrt',  # lower than default
            random_state=42
        ),
        # Tightened Gradient Boosting model for prediction
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=50,  # increased
            max_depth=4,  # decreased
            min_samples_split=50,  # increased
            min_samples_leaf=10,  # increased
            max_features='sqrt',  # lower than default
            random_state=42
        ),
        # Also applying tightened parameters to Extra Trees for consistency
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=50,        # increased
            max_depth=4,            # decreased
            min_samples_split=50,   # increased
            min_samples_leaf=10,    # increased
            max_features='sqrt',    # lower than default
            random_state=42
        ),
        'MLP': MLPRegressor(random_state=42, max_iter=1000, tol=1e-4, early_stopping=True),
        'SVR': SVR(),
        'ZeroR': DummyRegressor(strategy='mean')
    }
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in local_models.items():
        cv_r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        cv_mse_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        test_vaf = 1 - np.var(y_test - y_pred) / np.var(y_test)

        results[name] = {
            "CV R² Mean (Synthetic)": np.mean(cv_r2_scores),
            "CV R² Std (Synthetic)": np.std(cv_r2_scores),
            "CV MSE Mean (Synthetic)": np.mean(cv_mse_scores),
            "Test R² (Original)": test_r2,
            "Test MSE (Original)": test_mse,
            "Test VAF (Original)": test_vaf
        }
    return results


# =============================
# 6. Print Performance Metrics & Collect Predictions
# =============================
# Global models dictionary used for predictions (and later for plotting/export)
models = {
    'Ridge': Ridge(alpha=0.5),
    'Lasso': Lasso(alpha=0.01, max_iter=1000),
    'ElasticNet': ElasticNet(random_state=42),
    # Tightened RF model for prediction
    'Random Forest': RandomForestRegressor(
        n_estimators=50,  # increased
        max_depth=4,  # decreased
        min_samples_split=50,  # increased
        min_samples_leaf=10,  # increased
        max_features='sqrt',  # lower than default
        random_state=42
    ),
    # Tightened Gradient Boosting model for prediction
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=50,  # increased
        max_depth=4,  # decreased
        min_samples_split=50,  # increased
        min_samples_leaf=10,  # increased
        max_features='sqrt',  # lower than default
        random_state=42
    ),
    # Tightened Extra Trees model for prediction
    'Extra Trees': ExtraTreesRegressor(
        n_estimators=50,  # increased
        max_depth=4,  # decreased
        min_samples_split=50,  # increased
        min_samples_leaf=10,  # increased
        max_features='sqrt',  # lower than default
        random_state=42
    ),
    'MLP': MLPRegressor(random_state=42, max_iter=1000, tol=1e-4, early_stopping=True),
    'SVR': SVR(),
    'ZeroR': DummyRegressor(strategy='mean')
}

# Dictionaries to store performance metrics and predictions
performance_dict = {}
predictions_dict = {}

for method_name, synthesize_func in synthetic_methods.items():
    print(f"\n========== Training on Synthetic Data ({method_name}) ==========")
    X_synthetic, y_synthetic = synthesize_func(X_poly, y)

    # Evaluate performance metrics for this synthetic method
    metrics = evaluate_models(X_synthetic, y_synthetic, X_poly, y)
    for model_name, metric_values in metrics.items():
        print(f"\n{model_name} Performance ({method_name}):")
        for metric, value in metric_values.items():
            print(f"{metric}: {value:.4f}")
        # Store performance metrics
        performance_dict.setdefault(model_name, {})[method_name] = metric_values

    # Collect predictions for each model (trained on the same synthetic data)
    for model_name, model in models.items():
        model_copy = deepcopy(model)
        model_copy.fit(X_synthetic, y_synthetic)
        y_pred = model_copy.predict(X_poly)
        predictions_dict.setdefault(model_name, {})[method_name] = y_pred

# =============================
# 7. Plotting Functions
# =============================
def plot_actual_vs_predicted(y_actual, y_pred, model_name, method_name):
    r2_value = r2_score(y_actual, y_pred)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_actual, y_pred, alpha=0.7, label=f"{model_name} Predictions")
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Fit")
    plt.xlabel("Actual ICR")
    plt.ylabel("Predicted ICR")
    plt.title(f"{model_name} Prediction vs Actual ICR ({method_name})\nR² = {r2_value:.4f}")
    plt.legend()
    plt.show()

def plot_residuals(y_actual, y_pred, model_name, method_name):
    residuals = y_actual - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, label='Residuals')
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='dashed', label='Zero Error')
    plt.xlabel("Predicted ICR")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residual Plot: {model_name} ({method_name})")
    plt.legend()
    plt.show()

def plot_qq(residuals, model_name, method_name):
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of Residuals\n{model_name} ({method_name})")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Ordered Residuals")
    plt.show()

# =============================
# 8. Generate Plots Using Stored Predictions
# =============================
for method_name in synthetic_methods.keys():
    for model_name in models.keys():
        y_pred = predictions_dict[model_name][method_name]
        # Plot Actual vs Predicted using stored predictions
        plot_actual_vs_predicted(y, y_pred, model_name, method_name)
        # Plot Residuals
        plot_residuals(y, y_pred, model_name, method_name)
        # Compute and plot Q-Q plot for residuals
        residuals = y - y_pred
        plot_qq(residuals, model_name, method_name)

# =============================
# 9. Export Prediction Tables to Excel
# =============================
pred_output_file = "Model_prediction_values.xlsx"
with pd.ExcelWriter(pred_output_file, engine='openpyxl') as writer:
    for model_name, preds in predictions_dict.items():
        df_pred = pd.DataFrame({"ICR actual": y[:28]})
        for method_name in synthetic_methods.keys():
            col_name = f"ICR predicted ({method_name})"
            df_pred[col_name] = preds[method_name][:28]
        sheet_name = model_name[:31]  # Excel sheet name limitation
        df_pred.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"\nExported prediction table for {model_name} to sheet {sheet_name}.")

print(f"\nAll prediction tables have been exported to {pred_output_file}.")
