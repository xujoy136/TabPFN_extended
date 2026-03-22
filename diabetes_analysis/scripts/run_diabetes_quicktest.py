#!/usr/bin/env python
"""
快速测试 Diabetes 数据分析的核心功能
在运行完整脚本之前用于验证环境

Usage:
    python scripts/run_diabetes_quicktest.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')

print("=" * 70)
print("Diabetes 数据分析 - 快速测试")
print("=" * 70)

# 1. 数据加载
print("\n1. 加载数据...")
# 获取脚本所在目录，然后计算数据集路径
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
data_path = project_root / 'dataset' / 'diabetes_dataset.csv'

if not data_path.exists():
    print(f"✗ 数据文件未找到: {data_path}")
    exit(1)

df = pd.read_csv(data_path)
print(f"   ✓ 数据集形状: {df.shape}")
print(f"   ✓ 列数: {len(df.columns)}")
print(f"   ✓ 列名: {list(df.columns)}")

# 2. 数据预处理
print("\n2. 数据预处理...")
y = df['target']
X = df.drop(columns=['target'])

print(f"   ✓ 特征矩阵: {X.shape}")
print(f"   ✓ 目标变量范围: [{y.min():.2f}, {y.max():.2f}]")
print(f"   ✓ 目标变量均值: {y.mean():.2f}")

# 3. 数据划分
print("\n3. 划分训练/测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"   ✓ 训练集: {len(X_train)} 样本")
print(f"   ✓ 测试集: {len(X_test)} 样本")

# 4. 测试 TabPFN
print("\n4. 测试 TabPFN Regressor...")
try:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, n_estimators=2, device='cpu'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"   ✓ TabPFN R²: {r2:.4f}")
    print(f"   ✓ TabPFN RMSE: {rmse:.2f}")
except Exception as e:
    print(f"   ✗ TabPFN 测试失败: {e}")

# 5. 测试其他模型
print("\n5. 测试其他回归模型...")
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

models_to_test = {
    'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
    'LinearRegression': LinearRegression()
}

for name, model in models_to_test.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"   ✓ {name}: R²={r2:.4f}")
    except Exception as e:
        print(f"   ✗ {name}: {e}")

# 6. 测试 SHAP
print("\n6. 测试 SHAP...")
try:
    import shap
    import matplotlib.pyplot as plt
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, n_estimators=2, device='cpu'
    )
    model.fit(X_train, y_train)
    
    # 使用小样本快速测试
    background = X_train.iloc[:10]
    test_sample = X_test.iloc[:5]
    
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(test_sample, nsamples=20)
    
    print(f"   ✓ SHAP 值形状: {shap_values.shape}")
    print(f"   ✓ 计算成功!")
    
except Exception as e:
    print(f"   ✗ SHAP 测试失败: {e}")

print("\n" + "=" * 70)
print("✅ 快速测试完成! 环境正常，可以运行完整分析")
print("=" * 70)
print("\n下一步:")
print("  python scripts/run_diabetes_full.py")
