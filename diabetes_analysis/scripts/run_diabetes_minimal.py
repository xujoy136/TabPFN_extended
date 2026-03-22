#!/usr/bin/env python
"""Diabetes 最小化分析 - 仅 TabPFN vs LinearRegression vs RandomForest"""

import os
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

print("="*60)
print("Diabetes 快速回归分析")
print("="*60)

# 加载数据
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
df = pd.read_csv(project_root / 'dataset' / 'diabetes_dataset.csv')

y = df['target']
X = df.drop(columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\n数据集: {X.shape}, 训练集: {len(X_train)}, 测试集: {len(X_test)}")

# 只对比 3 个模型
models = {
    'TabPFN': TabPFNRegressor.create_default_for_version(ModelVersion.V2, n_estimators=2, device='cuda'),
    'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
    'LinearRegression': LinearRegression(),
}

results = []
print("\n模型训练与评估:")
for name, model in models.items():
    print(f"  {name}...", end=" ", flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({'Model': name, 'R2': r2, 'RMSE': rmse})
    print(f"R²={r2:.4f}, RMSE={rmse:.2f}")

# 生成简单对比图
results_df = pd.DataFrame(results).sort_values('R2', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2196F3', '#90A4AE', '#90A4AE']
bars = ax.barh(results_df['Model'], results_df['R2'], color=colors)
ax.set_xlabel('R² Score')
ax.set_title('Diabetes Dataset - Model Comparison', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
for bar, val in zip(bars, results_df['R2']):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'diabetes_quick_comparison.png'), dpi=150)
plt.close()

# 保存结果
results_df.to_csv(os.path.join(OUTPUT_DIR, 'diabetes_quick_results.csv'), index=False)

print("\n" + "="*60)
print("结果:")
for i, row in results_df.iterrows():
    marker = "⭐" if row['Model'] == 'TabPFN' else "  "
    print(f"  {marker} {row['Model']}: R² = {row['R2']:.4f}")
print("\n✅ 完成! 结果保存在 diabetes_analysis/results/")
print("="*60)
