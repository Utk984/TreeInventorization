#!/usr/bin/env python3
"""
Depth Calibration Script - Linear vs Non-Linear Approaches

This script calibrates the predicted depth values using Google depth as ground truth.
It compares multiple approaches:
1. Simple Linear: google_depth = slope * pred_depth + intercept
2. Multiple Linear: google_depth = slope1 * pred_depth + slope2 * image_x + slope3 * image_y + intercept
3. Polynomial Features: Including quadratic terms and interactions
4. Random Forest: Tree-based non-linear approach
5. SVR with RBF kernel: Support Vector Regression with non-linear kernel

Steps:
1. Load the depth comparison data
2. Filter out invalid Google depth values (-1.0)
3. Train and evaluate all models
4. Compare performance and create visualizations
5. Save the best performing model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from pathlib import Path
import joblib
import seaborn as sns
import warnings
import logging
import time

warnings.filterwarnings('ignore')

# Configure logger for depth calibration
logger = logging.getLogger(__name__)


def load_and_clean_data(csv_path):
    """Load depth data and remove invalid entries."""
    logger.info(f"📊 Loading depth calibration data from: {csv_path}")
    start_time = time.time()
    
    df = pd.read_csv(csv_path)
    
    logger.info(f"📈 Original data points: {len(df)}")
    
    # Remove invalid Google depth values (-1.0 indicates no depth data)
    df_clean = df[df['google_depth'] != -1.0].copy()
    
    load_time = time.time() - start_time
    logger.info(f"✅ Data loaded and cleaned in {load_time:.3f}s")
    logger.info(f"📊 Valid data points: {len(df_clean)} (removed {len(df) - len(df_clean)} invalid)")
    
    return df_clean


def analyze_data_distribution(df):
    """Analyze the distribution of features."""
    logger.info("📊 Analyzing data distribution")
    
    for col in ['pred_depth', 'google_depth', 'image_x', 'image_y']:
        logger.info(f"📈 {col:12} - Mean: {df[col].mean():8.2f}, Std: {df[col].std():8.2f}, "
                   f"Min: {df[col].min():8.2f}, Max: {df[col].max():8.2f}")


def train_simple_linear(df):
    """Train simple linear regression: google_depth = slope * pred_depth + intercept"""
    logger.debug("🔄 Training Simple Linear Regression model")
    start_time = time.time()
    
    X = df[['pred_depth']].values
    y = df['google_depth'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    train_time = time.time() - start_time
    logger.info(f"✅ Simple Linear trained in {train_time:.3f}s - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    
    return {
        'name': 'Simple Linear',
        'model': model,
        'predictions': y_pred,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'equation': f"google_depth = {model.coef_[0]:.6f} * pred_depth + {model.intercept_:.6f}"
    }


def train_multiple_linear(df):
    """Train multiple linear regression with spatial coordinates."""
    X = df[['pred_depth', 'image_x', 'image_y']].values
    y = df['google_depth'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    return {
        'name': 'Multiple Linear',
        'model': model,
        'predictions': y_pred,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'equation': f"google_depth = {model.coef_[0]:.6f} * pred_depth + {model.coef_[1]:.6f} * image_x + {model.coef_[2]:.6f} * image_y + {model.intercept_:.6f}"
    }


def train_polynomial(df, degree=2):
    """Train polynomial regression with interaction terms."""
    X = df[['pred_depth', 'image_x', 'image_y']].values
    y = df['google_depth'].values
    
    # Create polynomial features
    poly_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    
    poly_pipeline.fit(X, y)
    
    y_pred = poly_pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(poly_pipeline, X, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    return {
        'name': f'Polynomial (degree={degree})',
        'model': poly_pipeline,
        'predictions': y_pred,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'equation': f"Polynomial features with degree {degree}"
    }


def train_random_forest(df):
    """Train Random Forest regression."""
    X = df[['pred_depth', 'image_x', 'image_y']].values
    y = df['google_depth'].values
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_names = ['pred_depth', 'image_x', 'image_y']
    
    return {
        'name': 'Random Forest',
        'model': model,
        'predictions': y_pred,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'equation': f"Tree-based ensemble model",
        'feature_importance': dict(zip(feature_names, feature_importance))
    }


def train_svr_rbf(df):
    """Train Support Vector Regression with RBF kernel."""
    X = df[['pred_depth', 'image_x', 'image_y']].values
    y = df['google_depth'].values
    
    # Scale features for SVR
    svr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1))
    ])
    
    svr_pipeline.fit(X, y)
    
    y_pred = svr_pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(svr_pipeline, X, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    return {
        'name': 'SVR (RBF)',
        'model': svr_pipeline,
        'predictions': y_pred,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'cv_r2_mean': cv_mean,
        'cv_r2_std': cv_std,
        'equation': f"Support Vector Regression with RBF kernel"
    }


def compare_models(df):
    """Train and compare all models."""
    logger.info("🔄 Training and comparing all calibration models")
    comparison_start_time = time.time()
    
    models = []
    
    # Train all models
    logger.info("🔄 Training Simple Linear...")
    models.append(train_simple_linear(df))
    
    logger.info("🔄 Training Multiple Linear...")
    models.append(train_multiple_linear(df))
    
    logger.info("🔄 Training Polynomial (degree=2)...")
    models.append(train_polynomial(df, degree=2))
    
    logger.info("🔄 Training Polynomial (degree=3)...")
    models.append(train_polynomial(df, degree=3))
    
    logger.info("🔄 Training Random Forest...")
    models.append(train_random_forest(df))
    
    logger.info("🔄 Training SVR with RBF kernel...")
    models.append(train_svr_rbf(df))
    
    # Sort by R² score
    models.sort(key=lambda x: x['r2'], reverse=True)
    
    comparison_time = time.time() - comparison_start_time
    logger.info(f"✅ Model comparison completed in {comparison_time:.3f}s")
    logger.info("📊 MODEL COMPARISON RESULTS")
    logger.info(f"{'Model':<20} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'CV R² (mean±std)':<20}")
    
    for model in models:
        logger.info(f"{model['name']:<20} {model['r2']:<8.4f} {model['rmse']:<8.4f} {model['mae']:<8.4f} {model['cv_r2_mean']:.4f}±{model['cv_r2_std']:.4f}")
    
    return models


def create_comprehensive_plots(df, models):
    """Create comprehensive visualization plots."""
    
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Model comparison bar chart
    ax1 = plt.subplot(3, 3, 1)
    model_names = [m['name'] for m in models]
    r2_scores = [m['r2'] for m in models]
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    bars = ax1.bar(range(len(models)), r2_scores, color=colors, alpha=0.7)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cross-validation comparison
    ax2 = plt.subplot(3, 3, 2)
    cv_means = [m['cv_r2_mean'] for m in models]
    cv_stds = [m['cv_r2_std'] for m in models]
    
    bars = ax2.bar(range(len(models)), cv_means, yerr=cv_stds, 
                   color=colors, alpha=0.7, capsize=5)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Cross-Validation R² Score')
    ax2.set_title('Cross-Validation Performance')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Predictions vs actual for best model
    ax3 = plt.subplot(3, 3, 3)
    best_model = models[0]
    ax3.scatter(df['google_depth'], best_model['predictions'], alpha=0.6, s=30)
    min_val = min(df['google_depth'].min(), best_model['predictions'].min())
    max_val = max(df['google_depth'].max(), best_model['predictions'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    ax3.set_xlabel('Actual Google Depth (meters)')
    ax3.set_ylabel('Predicted Depth (meters)')
    ax3.set_title(f'Best Model: {best_model["name"]} (R²={best_model["r2"]:.4f})')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residuals for best model
    ax4 = plt.subplot(3, 3, 4)
    residuals = df['google_depth'] - best_model['predictions']
    ax4.scatter(best_model['predictions'], residuals, alpha=0.6, s=30)
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Predicted Depth (meters)')
    ax4.set_ylabel('Residuals (meters)')
    ax4.set_title(f'Residuals: {best_model["name"]}')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: RMSE comparison
    ax5 = plt.subplot(3, 3, 5)
    rmse_scores = [m['rmse'] for m in models]
    bars = ax5.bar(range(len(models)), rmse_scores, color=colors, alpha=0.7)
    ax5.set_xlabel('Models')
    ax5.set_ylabel('RMSE (meters)')
    ax5.set_title('RMSE Comparison')
    ax5.set_xticks(range(len(models)))
    ax5.set_xticklabels(model_names, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Feature importance for Random Forest
    ax6 = plt.subplot(3, 3, 6)
    rf_model = next((m for m in models if m['name'] == 'Random Forest'), None)
    if rf_model and 'feature_importance' in rf_model:
        features = list(rf_model['feature_importance'].keys())
        importance = list(rf_model['feature_importance'].values())
        bars = ax6.bar(features, importance, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax6.set_ylabel('Feature Importance')
        ax6.set_title('Random Forest Feature Importance')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, importance):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{imp:.3f}', ha='center', va='bottom')
    
    # Plot 7: Residuals vs image_x for best model
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(df['image_x'], residuals, alpha=0.6, s=30)
    ax7.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax7.set_xlabel('Image X Coordinate')
    ax7.set_ylabel('Residuals (meters)')
    ax7.set_title('Residuals vs X Position')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Residuals vs image_y for best model
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(df['image_y'], residuals, alpha=0.6, s=30)
    ax8.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax8.set_xlabel('Image Y Coordinate')
    ax8.set_ylabel('Residuals (meters)')
    ax8.set_title('Residuals vs Y Position')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: MAE comparison
    ax9 = plt.subplot(3, 3, 9)
    mae_scores = [m['mae'] for m in models]
    bars = ax9.bar(range(len(models)), mae_scores, color=colors, alpha=0.7)
    ax9.set_xlabel('Models')
    ax9.set_ylabel('MAE (meters)')
    ax9.set_title('MAE Comparison')
    ax9.set_xticks(range(len(models)))
    ax9.set_xticklabels(model_names, rotation=45, ha='right')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eval/depth_calibration_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"📊 Comprehensive comparison plots saved to eval/depth_calibration_comparison.png")


def save_best_model(df, models, output_path):
    """Save the best performing model and results."""
    best_model = models[0]
    
    # Apply best model calibration
    if best_model['name'] == 'Simple Linear':
        X = df[['pred_depth']].values
    else:
        X = df[['pred_depth', 'image_x', 'image_y']].values
    
    df['calibrated_pred_depth'] = best_model['predictions']
    
    # Save calibrated data
    df.to_csv(output_path, index=False)
    
    # Save best model
    model_path = output_path.parent / 'depth_calibration_best_model.pkl'
    joblib.dump(best_model['model'], model_path)
    
    # Save detailed results
    results_path = output_path.parent / 'calibration_results.txt'
    with open(results_path, 'w') as f:
        f.write(f"Depth Calibration Results - Model Comparison\n")
        f.write(f"=============================================\n\n")
        
        f.write(f"BEST MODEL: {best_model['name']}\n")
        f.write(f"R² Score: {best_model['r2']:.6f}\n")
        f.write(f"RMSE: {best_model['rmse']:.6f} meters\n")
        f.write(f"MAE: {best_model['mae']:.6f} meters\n")
        f.write(f"Cross-validation R²: {best_model['cv_r2_mean']:.6f} ± {best_model['cv_r2_std']:.6f}\n")
        f.write(f"Equation: {best_model['equation']}\n\n")
        
        f.write(f"ALL MODELS COMPARISON:\n")
        f.write(f"{'Model':<20} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'CV R² (mean±std)':<20}\n")
        f.write("-" * 80 + "\n")
        
        for model in models:
            f.write(f"{model['name']:<20} {model['r2']:<8.4f} {model['rmse']:<8.4f} {model['mae']:<8.4f} {model['cv_r2_mean']:.4f}±{model['cv_r2_std']:.4f}\n")
    
    logger.info(f"💾 Best model results saved to: {output_path}")
    logger.info(f"💾 Best model saved to: {model_path}")
    logger.info(f"💾 Detailed results saved to: {results_path}")


def main():
    logger.info("🚀 Starting depth calibration pipeline")
    pipeline_start_time = time.time()
    
    input_path = Path("eval/tree_depth_values.csv")
    output_path = Path("eval/tree_depth_calibrated_best.csv")
    
    if not input_path.exists():
        logger.error(f"❌ Input file not found: {input_path}. Run extract_depth_values.py first.")
        return
    
    # Load and clean data
    df = load_and_clean_data(input_path)
    
    if len(df) < 10:
        logger.error("❌ Not enough valid data points for regression.")
        return
    
    # Analyze data distribution
    analyze_data_distribution(df)
    
    # Compare all models
    models = compare_models(df)
    
    # Create comprehensive plots
    logger.info("📊 Creating comprehensive visualization plots")
    create_comprehensive_plots(df, models)
    
    # Save best model and results
    logger.info("💾 Saving best model and results")
    save_best_model(df, models, output_path)
    
    # Print final summary
    best_model = models[0]
    total_time = time.time() - pipeline_start_time
    
    logger.info("🎉 DEPTH CALIBRATION PIPELINE COMPLETED")
    logger.info(f"⏱️ Total pipeline time: {total_time:.3f}s")
    logger.info(f"🏆 Best performing model: {best_model['name']}")
    logger.info(f"📊 R² Score: {best_model['r2']:.4f}")
    logger.info(f"📏 RMSE: {best_model['rmse']:.4f} meters")
    logger.info(f"📏 MAE: {best_model['mae']:.4f} meters")
    logger.info(f"🔄 Cross-validation R²: {best_model['cv_r2_mean']:.4f} ± {best_model['cv_r2_std']:.4f}")
    
    if 'feature_importance' in best_model:
        logger.info("🔍 Feature Importance:")
        for feature, importance in best_model['feature_importance'].items():
            logger.info(f"  {feature}: {importance:.4f}")


if __name__ == "__main__":
    main() 