# Bike Sharing Demand Regression - 5 models comparison using Normal Equation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# ============================================================================
# REGRESSION FUNCTIONS
# ============================================================================
# //“This block computes θ by solving the normal equation
# using a bias-augmented design matrix.”
def fit_normal_equation(X, y):
    """Fit linear regression using Normal Equation: theta = (X^T X)^(-1) X^T y"""
    ones = np.ones((X.shape[0], 1))
    X_bias = np.hstack([ones, X])
    
    XtX = X_bias.T @ X_bias
    Xty = X_bias.T @ y
    theta = np.linalg.solve(XtX, Xty)
    
    return theta

# X*theta
def predict(X, theta):
    """Make predictions using trained parameters"""
    ones = np.ones((X.shape[0], 1))
    X_bias = np.hstack([ones, X])
    return X_bias @ theta


# make n degree poynomial matrix
def transform_polynomial(X, degree):
    """Transform to polynomial features without interactions"""
    X_poly = X.copy()
    for d in range(2, degree + 1):
        X_poly = np.hstack([X_poly, X ** d])
    return X_poly


def transform_quadratic_interactions(X):
    """Transform to quadratic features WITH interactions"""
    n_features = X.shape[1]
    X_transformed = X.copy()
    
    # Add squared terms
    X_transformed = np.hstack([X_transformed, X ** 2])
    
    # Add interaction terms
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
            X_transformed = np.hstack([X_transformed, interaction])
    
    return X_transformed


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# higher is better prediction 
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def adjusted_r2_score(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n - n_features - 1 <= 0:
        return r2
    return 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))


def evaluate_model(theta, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model on train and test sets"""
    y_train_pred = predict(X_train, theta)
    y_test_pred = predict(X_test, theta)
    
    return {
        'model_name': model_name,
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_rmse': root_mean_squared_error(y_train, y_train_pred),
        'test_rmse': root_mean_squared_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
    }


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def engineer_features(df):
    """Create new features from datetime and raw data"""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dayofyear'] = df['datetime'].dt.dayofyear
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Derived features
#     What it does
# Converts Saturday and Sunday into 1
# Converts weekdays (Mon–Fri) into 0
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                           (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    
    df = df.drop('datetime', axis=1)
    return df

# “This function separates features and target while 
# removing leakage variables to ensure valid regression training.”
def prepare_features_target(df, target_col='count'):
    """Separate X and y, remove target and leakage columns"""
    df = df.copy()
    drop_cols = [target_col]
    
    if 'casual' in df.columns:
        drop_cols.append('casual')
    if 'registered' in df.columns:
        drop_cols.append('registered')
    
    y = df[target_col].values if target_col in df.columns else None
    X = df.drop(drop_cols, axis=1, errors='ignore')
    
    return X.values, y
# “By giving all features zero mean and unit variance, 
# standardization prevents large-scale features from dominating the optimization, 
# ensuring fair contribution from each feature.”
def standardize_features(X_train, X_test):
    """Apply z-score normalization (fit on train only to prevent data leakage)"""
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std[train_std == 0] = 1.0
    
    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std
    
    return X_train_scaled, X_test_scaled


def train_test_split(X, y, test_size=0.2, random_state=42):
#     test_size=0.2 means 20% data for testing
# random_state=42 ensures reproducibility
    """Split data into train and test sets randomly"""
    np.random.seed(random_state)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(results_df, save_path='model_comparison.png'):
    """Create 4-subplot comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison - Bike Sharing Demand Regression', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    results_df.plot(x='model_name', y='test_mse', kind='bar', ax=ax1, color='steelblue', legend=False)
    ax1.set_title('Test Set MSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Mean Squared Error', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.2f', padding=3)
    
    ax2 = axes[0, 1]
    results_df.plot(x='model_name', y='test_r2', kind='bar', ax=ax2, color='forestgreen', legend=False)
    ax2.set_title('Test Set R² Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Model', fontsize=11)
    ax2.set_ylabel('R² Score', fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.4f', padding=3)
    
    ax3 = axes[1, 0]
    x_pos = np.arange(len(results_df))
    width = 0.35
    ax3.bar(x_pos - width/2, results_df['train_r2'], width, label='Train R²', color='lightcoral')
    ax3.bar(x_pos + width/2, results_df['test_r2'], width, label='Test R²', color='lightblue')
    ax3.set_title('Train vs Test R² (Overfitting Analysis)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Model', fontsize=11)
    ax3.set_ylabel('R² Score', fontsize=11)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(results_df['model_name'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1])
    
    ax4 = axes[1, 1]
    results_df.plot(x='model_name', y='test_rmse', kind='bar', ax=ax4, color='coral', legend=False)
    ax4.set_title('Test Set RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Model', fontsize=11)
    ax4.set_ylabel('Root Mean Squared Error', fontsize=11)
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.show()


def plot_predictions_vs_actual(y_test, predictions_dict, save_path='predictions_comparison.png'):
    """Create scatter plots: predicted vs actual for each model"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    fig.suptitle('Predicted vs Actual: All Models', fontsize=16, fontweight='bold')
    
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, edgecolors='k', linewidth=0.5)
        
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        r2 = r2_score(y_test, y_pred)
        ax.set_xlabel('Actual Count', fontsize=10)
        ax.set_ylabel('Predicted Count', fontsize=10)
        ax.set_title(f'{model_name}\nR² = {r2:.4f}', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    for idx in range(len(predictions_dict), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions comparison saved to: {save_path}")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("BIKE SHARING DEMAND - REGRESSION ANALYSIS")
    
    # Load and preprocess data
    df = pd.read_csv('train.csv')
    df_engineered = engineer_features(df)
    X, y = prepare_features_target(df_engineered, target_col='count')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled = standardize_features(X_train, X_test)
    
    print("Training models...\n")
    results = []
    predictions = {}
    
    # Model 1: Linear Regression
    theta_lr = fit_normal_equation(X_train_scaled, y_train)
    result_lr = evaluate_model(theta_lr, X_train_scaled, y_train, X_test_scaled, y_test, "Linear Regression")
    results.append(result_lr)
    predictions['Linear Regression'] = predict(X_test_scaled, theta_lr)
    
    # Model 2-4: Polynomial Regression (d=2, 3, 4)
    for degree in [2, 3, 4]:
        X_train_poly = transform_polynomial(X_train_scaled, degree)
        X_test_poly = transform_polynomial(X_test_scaled, degree)
        
        theta_poly = fit_normal_equation(X_train_poly, y_train)
        result_poly = evaluate_model(theta_poly, X_train_poly, y_train, X_test_poly, y_test, f"Polynomial (d={degree})")
        results.append(result_poly)
        predictions[f'Polynomial (d={degree})'] = predict(X_test_poly, theta_poly)
    
    # Model 5: Quadratic with Interactions
    X_train_interact = transform_quadratic_interactions(X_train_scaled)
    X_test_interact = transform_quadratic_interactions(X_test_scaled)
    
    theta_interact = fit_normal_equation(X_train_interact, y_train)
    result_interact = evaluate_model(theta_interact, X_train_interact, y_train, X_test_interact, y_test, "Quadratic + Interactions")
    results.append(result_interact)
    predictions['Quadratic + Interactions'] = predict(X_test_interact, theta_interact)
    
    # Results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_results.csv', index=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print("\n" + results_df.to_string(index=False))
    
    # Best model
    best_idx = results_df['test_r2'].idxmax()
    best_name = results_df.loc[best_idx, 'model_name']
    best_r2 = results_df.loc[best_idx, 'test_r2']
    best_mse = results_df.loc[best_idx, 'test_mse']
    best_rmse = results_df.loc[best_idx, 'test_rmse']
    
    print(f"\n{'='*80}")
    print("BEST MODEL")
    print("="*80)
    print(f"Model: {best_name}")
    print(f"Test R²: {best_r2:.6f}")
    print(f"Test MSE: {best_mse:.4f}")
    print(f"Test RMSE: {best_rmse:.4f}")
    
    # Visualizations
    plot_results(results_df, save_path='model_comparison.png')
    plot_predictions_vs_actual(y_test, predictions, save_path='predictions_vs_actual.png')
    
    print(f"\n{'='*80}")
    print("Files saved: model_results.csv, model_comparison.png, predictions_vs_actual.png")
    print("="*80)
    
    return results_df, predictions, (X_train_scaled, X_test_scaled, y_train, y_test)


if __name__ == "__main__":
    results_df, predictions, data = main()
