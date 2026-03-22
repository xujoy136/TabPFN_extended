#!/usr/bin/env python
"""
快速测试 Pregenant 数据分析的核心功能
在运行完整 Notebook 之前用于验证环境

Usage:
    python run_preganent_quicktest.py
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')

print("=" * 70)
print("Pregenant 数据分析 - 快速测试")
print("=" * 70)

# 1. 数据加载
print("\n1. 加载数据...")
data_path = Path('../../dataset/preganent raw data 202603(1).csv')
if not data_path.exists():
    print(f"✗ 数据文件未找到: {data_path}")
    exit(1)

df = pd.read_csv(data_path)
print(f"   ✓ 数据集形状: {df.shape}")
print(f"   ✓ 列数: {len(df.columns)}")

# 2. 数据预处理
print("\n2. 数据预处理...")

def preprocess_data(df):
    data = df.copy()
    cols_to_drop = ['name', 'Annotation', 'EM score']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
    data['Pregent'] = data['Pregent'].map({'N': 0, 'Y': 1})
    data = data.dropna(subset=['Pregent'])
    y = data['Pregent'].astype(int)
    X = data.drop(columns=['Pregent'])
    X = X.replace('N/A', np.nan)
    X = X.replace('', np.nan)
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    X_log = np.log1p(X + 1e-10)
    return X_log, y

X, y = preprocess_data(df)
print(f"   ✓ 特征矩阵: {X.shape}")
print(f"   ✓ 目标分布: N={sum(y==0)}, Y={sum(y==1)}")

# 3. 数据划分
print("\n3. 划分训练/测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"   ✓ 训练集: {len(X_train)} 样本")
print(f"   ✓ 测试集: {len(X_test)} 样本")

# 4. 测试 TabPFN
print("\n4. 测试 TabPFN...")
try:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, n_estimators=2, device='cpu'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
    
    print(f"   ✓ TabPFN 准确率: {acc:.4f}")
    print(f"   ✓ TabPFN F1分数: {f1:.4f}")
    print(f"   ✓ TabPFN AUC: {auc:.4f}")
except Exception as e:
    print(f"   ✗ TabPFN 测试失败: {e}")

# 5. 测试其他模型
print("\n5. 测试其他模型 (简化)...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

models_to_test = {
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=500)
}

for name, model in models_to_test.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"   ✓ {name}: {acc:.4f}")
    except Exception as e:
        print(f"   ✗ {name}: {e}")

# 6. 测试 SHAP
print("\n6. 测试 SHAP...")
try:
    import shap
    import matplotlib.pyplot as plt
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, n_estimators=2, device='cpu'
    )
    model.fit(X_train, y_train)
    
    # 使用小样本快速测试
    background = X_train.iloc[:10]
    test_sample = X_test.iloc[:5]
    
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(test_sample, nsamples=20)
    
    # 处理 SHAP 值维度 (修复多维问题)
    shap_values_array = np.array(shap_values)
    if shap_values_array.ndim == 3:
        shap_values_class1 = shap_values_array[:, :, 1]  # 提取类别 1
    elif isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values
    
    print(f"   ✓ SHAP 原始形状: {shap_values_array.shape}")
    print(f"   ✓ SHAP 类别1形状: {shap_values_class1.shape}")
    
    # 测试 beeswarm 图
    explanation = shap.Explanation(
        values=shap_values_class1,
        data=test_sample.values,
        feature_names=X_train.columns.tolist()
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.beeswarm(explanation, max_display=10, show=False)
    plt.close()
    print(f"   ✓ SHAP beeswarm 图生成成功")
    
except Exception as e:
    print(f"   ✗ SHAP 测试失败: {e}")

print("\n" + "=" * 70)
print("✅ 快速测试完成! 环境正常，可以运行 Notebook")
print("=" * 70)
print("\n下一步:")
print("  jupyter notebook preganent_analysis.ipynb")
