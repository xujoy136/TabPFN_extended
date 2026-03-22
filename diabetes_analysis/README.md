# Diabetes 数据集回归分析项目

本项目使用 TabPFN 和其他机器学习回归模型对 Diabetes 数据集进行分析和 Benchmark 对比。

## 数据集说明

Diabetes 数据集包含 442 个糖尿病患者的生理数据，目标是预测一年后疾病进展的定量指标（连续值）。

- **样本数**: 442
- **特征数**: 10 (均为标准化后的生理指标)
- **目标变量**: 一年后疾病进展的定量测量值 (连续值)
- **特征列表**:
  - `age`: 年龄
  - `sex`: 性别
  - `bmi`: 体质指数
  - `bp`: 平均血压
  - `s1-s6`: 6个血清测量值

## 目录结构

```
diabetes_analysis/
├── scripts/                 # 可执行的分析脚本
│   ├── run_diabetes_full.py       # 完整回归分析
│   └── run_diabetes_quicktest.py  # 环境测试
├── results/                 # 分析结果输出目录
│   ├── *.png                      # 生成的图表
│   └── *.csv                      # 结果数据
└── docs/                    # 文档
    └── README.md
```

## 快速开始

### 1. 完整分析（推荐）

运行完整分析，包含所有回归模型对比和 SHAP 解释性分析：

```bash
cd /xuzhuo/LGLC/TabPFN
python diabetes_analysis/scripts/run_diabetes_full.py
```

### 2. 环境测试

在运行完整分析前，先测试环境是否正常：

```bash
cd /xuzhuo/LGLC/TabPFN
python diabetes_analysis/scripts/run_diabetes_quicktest.py
```

## 功能特性

- **多模型 Benchmark**: 
  - TabPFN Regressor
  - XGBoost
  - LightGBM
  - RandomForest
  - GradientBoosting
  - MLP
  - SVR
  - DecisionTree
  - LinearRegression
  - Ridge

- **回归评估指标**:
  - R² Score (决定系数)
  - RMSE (均方根误差)
  - MAE (平均绝对误差)
  - MSE (均方误差)

- **SHAP 解释性分析**: 
  - 特征重要性排序 (Feature Importance)
  - Beeswarm 图 (特征影响分布)
  - Dependence 图 (特征依赖关系)

- **交叉验证**: 5-Fold CV 评估
- **可视化**: 
  - 预测值 vs 真实值散点图
  - 残差图
  - 特征相关性热力图

## 输出文件

所有输出文件保存在 `results/` 目录：

| 文件 | 说明 |
|------|------|
| `diabetes_benchmark_comparison.png` | 多指标对比图 (R², RMSE, MAE, 训练时间) |
| `diabetes_prediction_scatter.png` | 预测值 vs 真实值散点图 |
| `diabetes_residual_plots.png` | 残差分析图 |
| `diabetes_shap_bar.png` | SHAP 特征重要性 (Bar) |
| `diabetes_shap_summary.png` | SHAP Summary Plot |
| `diabetes_shap_dependence.png` | SHAP Dependence Plot |
| `diabetes_correlation_heatmap.png` | 特征相关性热力图 |
| `diabetes_cv_results.png` | 交叉验证结果 |
| `*.csv` | 结果数据文件 |

## 依赖

- tabpfn
- shap
- xgboost
- lightgbm
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## 数据来源

数据集位于: `../../dataset/diabetes_dataset.csv`

原始数据来源: sklearn.datasets.load_diabetes()
