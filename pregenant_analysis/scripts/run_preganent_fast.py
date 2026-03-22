#!/usr/bin/env python
"""
Pregenant 代谢组学数据快速分析脚本
优化版：减少 SHAP 计算时间

Usage:
    python scripts/run_preganent_fast.py

输出文件保存在 results/ 目录
"""

import os

# 设置输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

warnings.filterwarnings('ignore')
plt.style.use('default')

def print_section(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

def preprocess_data(df):
    """预处理代谢组学数据"""
    data = df.copy()
    cols_to_drop = ['name', 'Annotation', 'EM score']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    data['Pregent'] = data['Pregent'].map({'N': 0, 'Y': 1})
    data = data.dropna(subset=['Pregent'])
    y = data['Pregent'].astype(int)
    X = data.drop(columns=['Pregent'])
    X = X.replace(['N/A', ''], np.nan)
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    X_log = np.log1p(X + 1e-10)
    return X_log, y, X.columns.tolist()

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """评估单个模型性能"""
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
    
    results = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan,
        'Fit_Time': fit_time,
        'Predict_Time': predict_time
    }
    
    return results, y_pred, y_proba

def main():
    print("="*70)
    print("PREGENANT 代谢组学数据分析 (快速版)")
    print("="*70)
    
    # 1. 数据加载
    print_section("1. 数据加载与预处理")
    data_path = '../../dataset/preganent raw data 202603(1).csv'
    df = pd.read_csv(data_path)
    print(f"数据集形状: {df.shape}")
    
    X, y, feature_names = preprocess_data(df)
    print(f"预处理后: X={X.shape}, 特征数={len(feature_names)}")
    print(f"类别分布: N={sum(y==0)}, Y={sum(y==1)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")
    
    # 2. Benchmark 对比
    print_section("2. Benchmark 对比 (8种方法)")
    
    models = {
        'TabPFN': TabPFNClassifier.create_default_for_version(ModelVersion.V2, n_estimators=2, device='cpu'),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results_list = []
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        print(f"[{name}] ", end="", flush=True)
        try:
            result, y_pred, y_proba = evaluate_model(name, model, X_train, X_test, y_train, y_test)
            results_list.append(result)
            predictions[name] = y_pred
            probabilities[name] = y_proba
            print(f"✓ Accuracy: {result['Accuracy']:.4f}")
        except Exception as e:
            print(f"✗ Error: {str(e)[:40]}")
    
    results_df = pd.DataFrame(results_list).sort_values('Accuracy', ascending=False)
    print("\nBenchmark 结果:")
    print(results_df[['Model', 'Accuracy', 'F1-Score', 'ROC-AUC']].to_string(index=False))
    
    # 3. 生成 Benchmark 对比图
    print("\n生成可视化图表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        sorted_df = results_df.sort_values(metric, ascending=True)
        bar_colors = ['#2196F3' if 'TabPFN' in m else '#90A4AE' for m in sorted_df['Model']]
        bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=bar_colors)
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xlim(0, 1.05)
        for bar, val in zip(bars, sorted_df[metric]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=8)
    
    # 训练时间
    ax = axes[1, 2]
    sorted_time = results_df.sort_values('Fit_Time', ascending=True)
    bar_colors = ['#2196F3' if 'TabPFN' in m else '#90A4AE' for m in sorted_time['Model']]
    ax.barh(sorted_time['Model'], sorted_time['Fit_Time'], color=bar_colors)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    
    plt.suptitle('Pregenant Dataset - Model Benchmark Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pregenant_benchmark_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ preganent_benchmark_comparison.png")
    
    # ROC 曲线
    fig, ax = plt.subplots(figsize=(10, 8))
    for name in results_df['Model']:
        if name in probabilities:
            fpr, tpr, _ = roc_curve(y_test, probabilities[name])
            auc_score = roc_auc_score(y_test, probabilities[name])
            color = '#2196F3' if 'TabPFN' in name else None
            linewidth = 3 if 'TabPFN' in name else 1.5
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})", color=color, linewidth=linewidth)
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'pregenant_roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ preganent_roc_curves.png")
    
    # 4. SHAP 分析 (优化版)
    print_section("3. SHAP 解释性分析 (优化版)")
    print("训练 TabPFN 并计算 SHAP 值...")
    
    tabpfn_model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, n_estimators=2, device='cpu')
    tabpfn_model.fit(X_train, y_train)
    
    # 减少样本数以加速
    background = X_train.iloc[:10]
    test_sample = X_test.iloc[:5]
    
    print("计算 SHAP (nsamples=30, 约1-2分钟)...")
    explainer = shap.KernelExplainer(tabpfn_model.predict_proba, background)
    shap_values = explainer.shap_values(test_sample, nsamples=30)
    
    # 处理 SHAP 值维度
    shap_values_array = np.array(shap_values)
    if shap_values_array.ndim == 3:
        shap_values_class1 = shap_values_array[:, :, 1]
    elif isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values
    
    print(f"✓ SHAP 值形状: {shap_values_class1.shape}")
    
    # SHAP Bar Plot
    print("生成 SHAP 图表...")
    fig, ax = plt.subplots(figsize=(10, 12))
    shap.summary_plot(shap_values_class1, test_sample, feature_names=feature_names,
                      plot_type='bar', max_display=20, show=False)
    plt.title('Top 20 Important Metabolites (Mean |SHAP|)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pregenant_shap_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ preganent_shap_bar.png")
    
    # 手动 Beeswarm Plot
    mean_shap = np.abs(shap_values_class1).mean(axis=0)
    top_indices = np.argsort(mean_shap)[-20:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    y_positions = np.arange(len(top_features))
    
    for i, (feature_idx, y_pos) in enumerate(zip(top_indices, y_positions)):
        shap_vals = shap_values_class1[:, feature_idx]
        feature_vals = test_sample.iloc[:, feature_idx].values
        colors = plt.cm.RdBu_r((feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min() + 1e-10))
        jitter = np.random.normal(0, 0.02, len(shap_vals))
        ax.scatter(shap_vals, [y_pos] * len(shap_vals) + jitter, c=colors, alpha=0.6, s=50, 
                   edgecolors='black', linewidth=0.5)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
    ax.set_title('SHAP Beeswarm - Top 20 Metabolites', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Feature Value (Low → High)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pregenant_shap_beeswarm.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ preganent_shap_beeswarm.png")
    
    # Dependence Plot
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': mean_shap}).sort_values('importance', ascending=False)
    top_2 = feature_importance.head(2)['feature'].tolist()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, feature in enumerate(top_2):
        feature_idx = feature_names.index(feature)
        shap_vals = shap_values_class1[:, feature_idx]
        feature_vals = test_sample.iloc[:, feature_idx].values
        scatter = axes[idx].scatter(feature_vals, shap_vals, c=feature_vals, cmap='RdBu_r', 
                                    alpha=0.6, s=100, edgecolors='black', linewidth=1)
        axes[idx].set_xlabel(f'{feature} Value', fontsize=11)
        axes[idx].set_ylabel('SHAP Value', fontsize=11)
        axes[idx].set_title(f'SHAP Dependence: {feature}', fontsize=12, fontweight='bold')
        axes[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[idx].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[idx], shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pregenant_shap_dependence.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ preganent_shap_dependence.png")
    
    print("\nTop 5 重要代谢物:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']}: {row['importance']:.4f}")
    
    # 5. 代谢物分布
    print_section("4. 代谢物分布分析")
    
    top_10 = feature_importance.head(10)['feature'].tolist()
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_10):
        ax = axes[idx]
        feature_data = df[[feature, 'Pregent']].copy()
        feature_data[feature] = pd.to_numeric(feature_data[feature], errors='coerce')
        feature_data = feature_data.dropna()
        
        box_data = [
            feature_data[feature_data['Pregent'] == 'N'][feature].values,
            feature_data[feature_data['Pregent'] == 'Y'][feature].values
        ]
        bp = ax.boxplot(box_data, labels=['N', 'Y'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel('Pregnant')
    
    plt.suptitle('Top 10 Metabolites Distribution by Group', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pregenant_metabolite_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ preganent_metabolite_distribution.png")
    
    # 6. 相关性热力图
    print("生成相关性热力图...")
    top_20 = feature_importance.head(20)['feature'].tolist()
    top_indices = [feature_names.index(f) for f in top_20]
    corr_matrix = X.iloc[:, top_indices].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(top_20)))
    ax.set_yticks(np.arange(len(top_20)))
    ax.set_xticklabels(top_20, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(top_20, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Correlation', rotation=270, labelpad=15)
    
    for i in range(len(top_20)):
        for j in range(len(top_20)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=6)
    
    ax.set_title('Correlation Matrix - Top 20 Important Metabolites', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pregenant_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ preganent_correlation_heatmap.png")
    
    # 7. 交叉验证
    print_section("5. 交叉验证")
    
    cv_results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"[{name}] ", end="", flush=True)
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            cv_results.append({'Model': name, 'CV_Mean': cv_scores.mean(), 'CV_Std': cv_scores.std()})
            print(f"CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception as e:
            print(f"✗ Error")
    
    cv_df = pd.DataFrame(cv_results).sort_values('CV_Mean', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    models_sorted = cv_df['Model'].tolist()
    means = cv_df['CV_Mean'].tolist()
    stds = cv_df['CV_Std'].tolist()
    colors = ['#2196F3' if 'TabPFN' in m else '#90A4AE' for m in models_sorted]
    
    ax.barh(models_sorted, means, xerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_xlabel('CV Accuracy')
    ax.set_title('5-Fold Cross Validation Accuracy (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pregenant_cv_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ preganent_cv_results.png")
    
    # 8. 保存结果
    print_section("6. 保存结果")
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'pregenant_benchmark_results.csv'), index=False)
    feature_importance.to_csv(os.path.join(OUTPUT_DIR, 'pregenant_feature_importance.csv'), index=False)
    cv_df.to_csv(os.path.join(OUTPUT_DIR, 'pregenant_cv_results.csv'), index=False)
    
    print("✓ preganent_benchmark_results.csv")
    print("✓ preganent_feature_importance.csv")
    print("✓ preganent_cv_results.csv")
    
    # 汇总
    print_section("分析完成!")
    print(f"总样本数: {len(df)}")
    print(f"特征数: {len(feature_names)}")
    print("\n🏆 Benchmark 前3名:")
    for i, (_, row) in enumerate(results_df.head(3).iterrows(), 1):
        marker = "⭐" if 'TabPFN' in row['Model'] else "  "
        print(f"  {marker} {i}. {row['Model']}: {row['Accuracy']:.4f}")
    
    print("\n📁 生成的图表文件 (results/):")
    print("  1. preganent_benchmark_comparison.png")
    print("  2. preganent_roc_curves.png")
    print("  3. preganent_shap_bar.png")
    print("  4. preganent_shap_beeswarm.png")
    print("  5. preganent_shap_dependence.png")
    print("  6. preganent_metabolite_distribution.png")
    print("  7. preganent_correlation_heatmap.png")
    print("  8. preganent_cv_results.png")
    print("\n✅ 所有分析完成!")

if __name__ == "__main__":
    main()
