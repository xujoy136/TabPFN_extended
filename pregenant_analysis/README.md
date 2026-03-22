# Pregenant 代谢组学数据分析项目

本项目使用 TabPFN 和其他机器学习模型对 Pregenant 代谢组学数据进行分析和 Benchmark 对比。

## 目录结构

```
pregenant_analysis/
├── scripts/           # 可执行的分析脚本
│   ├── run_preganent_full.py       # 完整分析（包含全部模型和 SHAP）
│   ├── run_preganent_fast.py       # 快速分析（减少 SHAP 计算时间）
│   └── run_preganent_quicktest.py  # 快速测试环境
│
├── shap_modules/      # SHAP 解释性分析模块
│   ├── shap_analysis.py            # SHAP 分析基础模块
│   └── shap_analysis_v2.py         # SHAP 分析高级模块（支持本地数据集）
│
├── notebooks/         # Jupyter Notebook
│   └── preganent_analysis.ipynb    # 交互式分析 Notebook
│
├── results/           # 分析结果输出目录
│   ├── *.png                       # 生成的图表
│   └── *.csv                       # 结果数据
│
└── docs/              # 文档
    ├── README.md                   # 项目说明
    └── SHAP_README.md              # SHAP 分析说明
```

## 快速开始

### 1. 完整分析（推荐）

运行完整分析，包含所有模型对比和 SHAP 解释性分析：

```bash
cd /xuzhuo/LGLC/TabPFN
python pregenant_analysis/scripts/run_preganent_full.py
```

### 2. 快速分析

如果 SHAP 计算时间过长，可以使用快速版本：

```bash
cd /xuzhuo/LGLC/TabPFN
python pregenant_analysis/scripts/run_preganent_fast.py
```

### 3. 环境测试

在运行完整分析前，先测试环境是否正常：

```bash
cd /xuzhuo/LGLC/TabPFN
python pregenant_analysis/scripts/run_preganent_quicktest.py
```

### 4. 交互式 Notebook

```bash
cd /xuzhuo/LGLC/TabPFN/pregenant_analysis
jupyter notebook notebooks/preganent_analysis.ipynb
```

## 功能特性

- **多模型 Benchmark**: TabPFN、XGBoost、LightGBM、RandomForest、GradientBoosting、MLP、SVM、DecisionTree、LogisticRegression
- **SHAP 解释性分析**: 特征重要性、Beeswarm 图、Dependence 图
- **交叉验证**: 5-Fold CV 评估
- **可视化**: ROC 曲线、特征分布、相关性热力图

## 输出文件

所有输出文件保存在 `results/` 目录：

| 文件 | 说明 |
|------|------|
| `pregenant_benchmark_comparison.png` | 多指标对比图 |
| `pregenant_roc_curves.png` | ROC 曲线对比 |
| `pregenant_shap_bar.png` | SHAP 特征重要性 |
| `pregenant_shap_beeswarm.png` | SHAP Beeswarm 图 |
| `pregenant_shap_dependence.png` | SHAP Dependence 图 |
| `pregenant_metabolite_distribution.png` | 代谢物分布箱线图 |
| `pregenant_cv_results.png` | 交叉验证结果 |
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

数据集位于: `../../dataset/preganent raw data 202603(1).csv`
