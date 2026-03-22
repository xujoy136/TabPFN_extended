#!/usr/bin/env python
"""Diabetes SHAP 分析 (独立脚本)"""
import os
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
import shap
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Diabetes SHAP 解释性分析")
print("="*60)

# 加载数据
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
df = pd.read_csv(project_root / 'dataset' / 'diabetes_dataset.csv')

y = df['target']
X = df.drop(columns=['target'])
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\n数据集: {X.shape}, 特征: {len(feature_names)}")
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# 训练 TabPFN
print("\n训练 TabPFN...")
model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, n_estimators=4, device='cuda')
model.fit(X_train, y_train)

# SHAP 分析 - 使用更少样本
print("\n计算 SHAP 值...")
background = X_train.iloc[:20]
shap_sample = X_test.iloc[:30]  # 减少到 30 个样本

print(f"背景样本: {len(background)}, SHAP样本: {len(shap_sample)}")

explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(shap_sample, nsamples=50)  # 减少 nsamples

print(f"✓ SHAP 值计算完成，形状: {shap_values.shape}")

# 生成 SHAP 图表
print("\n生成 SHAP 图表...")

# 1. SHAP Bar Plot
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, shap_sample, feature_names=feature_names,
                  plot_type='bar', max_display=10, show=False)
plt.title('Feature Importance (Mean |SHAP|) - TabPFN Regression', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'diabetes_shap_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ diabetes_shap_bar.png")

# 2. SHAP Summary (Beeswarm)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, shap_sample, feature_names=feature_names,
                  max_display=10, show=False)
plt.title('SHAP Feature Impact - TabPFN Regression', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'diabetes_shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ diabetes_shap_summary.png")

# 3. 计算特征重要性
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

# 保存特征重要性
feature_importance.to_csv(os.path.join(OUTPUT_DIR, 'diabetes_feature_importance.csv'), index=False)
print("✓ diabetes_feature_importance.csv")

# 打印 Top 5 特征
print("\nTop 5 重要特征:")
for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
    print(f"  {i}. {row['feature']}: {row['importance']:.4f}")

print("\n" + "="*60)
print("✅ SHAP 分析完成!")
print("="*60)
