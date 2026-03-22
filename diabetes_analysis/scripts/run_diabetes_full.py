#!/usr/bin/env python
"""
Diabetes 数据集完整回归分析脚本
使用 TabPFN 和其他回归模型进行 benchmark 对比和 SHAP 解释性分析

Usage:
    python scripts/run_diabetes_full.py

输出文件保存在 results/ 目录
"""

import os

# 设置输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')  # 非交互式后端

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

def print_section(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

def preprocess_data(df):
    """预处理 diabetes 数据"""
    # 分离特征和目标
    y = df['target']
    X = df.drop(columns=['target'])
    
    # 获取特征名
    feature_names = X.columns.tolist()
    
    print(f"特征数: {len(feature_names)}")
    print(f"特征列表: {feature_names}")
    print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")
    print(f"目标变量均值: {y.mean():.2f} ± {y.std():.2f}")
    
    return X, y, feature_names

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """评估单个回归模型性能"""
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    # 计算回归指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Fit_Time': fit_time,
        'Predict_Time': predict_time
    }
    
    return results, y_pred

def main():
    print("="*70)
    print("DIABETES 数据集回归分析")
    print("="*70)
    
    # 1. 数据加载
    print_section("1. 数据加载与预处理")
    # 获取脚本所在目录，然后计算数据集路径
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_path = project_root / 'dataset' / 'diabetes_dataset.csv'
    df = pd.read_csv(data_path)
    print(f"数据集形状: {df.shape}")
    
    X, y, feature_names = preprocess_data(df)
    print(f"\n预处理后: X={X.shape}, 特征数={len(feature_names)}")
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    
    # 2. Benchmark 对比
    print_section("2. Benchmark 对比 (10种回归方法)")
    
    models = {
        'TabPFN': TabPFNRegressor.create_default_for_version(ModelVersion.V2, n_estimators=4, device='cuda'),
        'XGBoost': XGBRegressor(random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(random_state=42, verbose=-1),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
    }
    
    results_list = []
    predictions = {}
    
    for name, model in models.items():
        print(f"[{name}] ", end="", flush=True)
        try:
            result, y_pred = evaluate_model(name, model, X_train, X_test, y_train, y_test)
            results_list.append(result)
            predictions[name] = y_pred
            print(f"✓ R²: {result['R2']:.4f}, RMSE: {result['RMSE']:.2f}")
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
    
    results_df = pd.DataFrame(results_list).sort_values('R2', ascending=False)
    print("\nBenchmark 结果 (按 R² 排序):")
    print(results_df[['Model', 'R2', 'RMSE', 'MAE']].to_string(index=False))
    
    # 3. 生成 Benchmark 对比图
    print("\n生成可视化图表...")
    
    # R2 对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R² Score
    ax = axes[0, 0]
    sorted_df = results_df.sort_values('R2', ascending=True)
    bar_colors = ['#2196F3' if 'TabPFN' in m else '#90A4AE' for m in sorted_df['Model']]
    bars = ax.barh(sorted_df['Model'], sorted_df['R2'], color=bar_colors)
    ax.set_xlabel('R² Score')
    ax.set_title('R² Score Comparison (Higher is Better)')
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, sorted_df['R2']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
    
    # RMSE
    ax = axes[0, 1]
    sorted_df = results_df.sort_values('RMSE', ascending=False)
    bar_colors = ['#2196F3' if 'TabPFN' in m else '#90A4AE' for m in sorted_df['Model']]
    bars = ax.barh(sorted_df['Model'], sorted_df['RMSE'], color=bar_colors)
    ax.set_xlabel('RMSE (Lower is Better)')
    ax.set_title('RMSE Comparison')
    for bar, val in zip(bars, sorted_df['RMSE']):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=9)
    
    # MAE
    ax = axes[1, 0]
    sorted_df = results_df.sort_values('MAE', ascending=False)
    bar_colors = ['#2196F3' if 'TabPFN' in m else '#90A4AE' for m in sorted_df['Model']]
    bars = ax.barh(sorted_df['Model'], sorted_df['MAE'], color=bar_colors)
    ax.set_xlabel('MAE (Lower is Better)')
    ax.set_title('MAE Comparison')
    for bar, val in zip(bars, sorted_df['MAE']):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=9)
    
    # 训练时间
    ax = axes[1, 1]
    sorted_time = results_df.sort_values('Fit_Time', ascending=True)
    bar_colors = ['#2196F3' if 'TabPFN' in m else '#90A4AE' for m in sorted_time['Model']]
    ax.barh(sorted_time['Model'], sorted_time['Fit_Time'], color=bar_colors)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    
    plt.suptitle('Diabetes Dataset - Regression Model Benchmark', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diabetes_benchmark_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ diabetes_benchmark_comparison.png")
    
    # 预测值 vs 真实值散点图
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        if idx >= 10:
            break
        name = row['Model']
        ax = axes[idx]
        y_pred = predictions[name]
        
        ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{name}\nR²={row["R2"]:.3f}, RMSE={row["RMSE"]:.1f}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Predicted vs True Values', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diabetes_prediction_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ diabetes_prediction_scatter.png")
    
    # 残差图
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        if idx >= 10:
            break
        name = row['Model']
        ax = axes[idx]
        y_pred = predictions[name]
        residuals = y_test - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{name} Residuals')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Residual Plots', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diabetes_residual_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ diabetes_residual_plots.png")
    
    # 4. SHAP 分析
    print_section("3. SHAP 解释性分析")
    print("训练 TabPFN 并计算 SHAP 值...")
    
    tabpfn_model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, n_estimators=4, device='cpu')
    tabpfn_model.fit(X_train, y_train)
    
    # 使用较小的背景数据
    background = X_train.iloc[:min(30, len(X_train))]
    shap_sample = X_test.iloc[:min(50, len(X_test))]  # 使用测试集的一部分
    
    print(f"背景样本: {len(background)}, SHAP样本: {len(shap_sample)}")
    print("计算 SHAP 值 (这可能需要几分钟)...")
    
    explainer = shap.KernelExplainer(tabpfn_model.predict, background)
    shap_values = explainer.shap_values(shap_sample, nsamples=100)
    
    print(f"SHAP 值形状: {shap_values.shape}")
    
    # SHAP Summary Plot (Bar)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_sample, feature_names=feature_names,
                      plot_type='bar', max_display=10, show=False)
    plt.title('Feature Importance (Mean |SHAP|) - TabPFN Regression', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diabetes_shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ diabetes_shap_bar.png")
    
    # SHAP Summary Plot (Beeswarm)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_sample, feature_names=feature_names,
                      max_display=10, show=False)
    plt.title('SHAP Feature Impact - TabPFN Regression', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diabetes_shap_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ diabetes_shap_summary.png")
    
    # 计算特征重要性
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    # Dependence Plot for top 2 features
    top_2 = feature_importance.head(2)['feature'].tolist()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, feature in enumerate(top_2):
        feature_idx = feature_names.index(feature)
        shap_vals = shap_values[:, feature_idx]
        feature_vals = shap_sample.iloc[:, feature_idx].values
        
        scatter = axes[idx].scatter(feature_vals, shap_vals, 
                                    c=feature_vals, cmap='viridis',
                                    alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        axes[idx].set_xlabel(f'{feature} Value', fontsize=11)
        axes[idx].set_ylabel('SHAP Value', fontsize=11)
        axes[idx].set_title(f'SHAP Dependence: {feature}', fontsize=12, fontweight='bold')
        axes[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[idx].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[idx], shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diabetes_shap_dependence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ diabetes_shap_dependence.png")
    
    print("\nTop 5 重要特征:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance']:.4f}")
    
    # 5. 特征相关性分析
    print_section("4. 特征相关性分析")
    
    # 相关性热力图
    corr_matrix = X.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    
    # 添加数值标注
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', rotation=270, labelpad=15)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diabetes_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ diabetes_correlation_heatmap.png")
    
    # 6. 交叉验证
    print_section("5. 交叉验证")
    
    cv_results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"[{name}] ", end="", flush=True)
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            cv_results.append({
                'Model': name, 
                'CV_R2_Mean': cv_scores.mean(), 
                'CV_R2_Std': cv_scores.std()
            })
            print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception as e:
            print(f"✗ Error")
    
    cv_df = pd.DataFrame(cv_results).sort_values('CV_R2_Mean', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    models_sorted = cv_df['Model'].tolist()
    means = cv_df['CV_R2_Mean'].tolist()
    stds = cv_df['CV_R2_Std'].tolist()
    colors = ['#2196F3' if 'TabPFN' in m else '#90A4AE' for m in models_sorted]
    
    ax.barh(models_sorted, means, xerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_xlabel('CV R² Score')
    ax.set_title('5-Fold Cross Validation R² (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'diabetes_cv_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ diabetes_cv_results.png")
    
    # 7. 保存结果
    print_section("6. 保存结果")
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'diabetes_benchmark_results.csv'), index=False)
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, 'diabetes_feature_importance.csv'), index=False)
    cv_df.to_csv(os.path.join(OUTPUT_DIR, 'diabetes_cv_results.csv'), index=False)
    
    print("✓ diabetes_benchmark_results.csv")
    print("✓ diabetes_feature_importance.csv")
    print("✓ diabetes_cv_results.csv")
    
    # 汇总
    print_section("分析完成!")
    print(f"总样本数: {len(df)}")
    print(f"特征数: {len(feature_names)}")
    print(f"\n目标变量统计:")
    print(f"  最小值: {y.min():.2f}")
    print(f"  最大值: {y.max():.2f}")
    print(f"  均值: {y.mean():.2f}")
    print(f"  标准差: {y.std():.2f}")
    
    print("\n🏆 Benchmark 前3名 (按 R²):")
    for i, (_, row) in enumerate(results_df.head(3).iterrows(), 1):
        marker = "⭐" if 'TabPFN' in row['Model'] else "  "
        print(f"  {marker} {i}. {row['Model']}: R²={row['R2']:.4f}, RMSE={row['RMSE']:.2f}")
    
    print("\n📁 生成的文件 (results/):")
    print("  - diabetes_benchmark_comparison.png")
    print("  - diabetes_prediction_scatter.png")
    print("  - diabetes_residual_plots.png")
    print("  - diabetes_shap_bar.png")
    print("  - diabetes_shap_summary.png")
    print("  - diabetes_shap_dependence.png")
    print("  - diabetes_correlation_heatmap.png")
    print("  - diabetes_cv_results.png")
    print("\n✅ 所有分析完成!")

if __name__ == "__main__":
    main()
