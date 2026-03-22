# Pregenant 代谢组学数据分析 Notebook

## 📋 概述

本 Jupyter Notebook 对孕妇代谢组学数据进行综合分析，结合 **Benchmark 对比** 和 **SHAP 解释性分析**。

## 📊 数据集信息

- **文件名**: `preganent raw data 202603(1).csv`
- **样本数**: 40
- **特征数**: 127 (代谢物浓度)
- **目标变量**: `Pregent` (N=0, Y=1)
- **忽略列**: `name`, `Annotation`, `EM score`

### 数据预处理
1. 移除不需要的列 (name, Annotation, EM score)
2. 目标变量编码: N→0, Y→1
3. 缺失值处理: 用中位数填充
4. 对数转换: log1p 处理偏态分布

## 🚀 使用方法

### 1. 启动 Jupyter

```bash
cd /xuzhuo/LGLC/TabPFN/benchmarks/notebooks
jupyter notebook preganent_analysis.ipynb
```

### 2. 按顺序运行 Cell

Notebook 包含以下主要部分：

#### **Part 1: 数据加载与预处理**
- 加载 CSV 数据
- 数据清洗和转换
- 训练集/测试集划分 (70/30)

#### **Part 2: Benchmark 对比 (9种方法)**
- TabPFN
- XGBoost
- LightGBM
- RandomForest
- GradientBoosting
- MLP (神经网络)
- SVM
- DecisionTree
- LogisticRegression

**评估指标**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- 训练/预测时间

#### **Part 3: SHAP 解释性分析**
- 基于 TabPFN 的 SHAP 值计算
- Beeswarm 图 (特征影响分布)
- Bar 图 (特征重要性排名)
- Dependence Plot (特征依赖关系)

#### **Part 4: 代谢物分布分析**
- Top 10 代谢物箱线图
- 相关性热力图

#### **Part 5: 交叉验证**
- 5-Fold Stratified CV
- 性能稳定性评估

## 📈 输出结果

运行完成后会生成以下文件：

### CSV 文件
| 文件名 | 内容 |
|--------|------|
| `pregenant_benchmark_results.csv` | 9种方法的 Benchmark 对比结果 |
| `pregenant_feature_importance.csv` | 127个代谢物的 SHAP 重要性排名 |
| `pregenant_cv_results.csv` | 5折交叉验证结果 |

### 可视化图表
| 文件名 | 描述 |
|--------|------|
| `pregenant_benchmark_comparison.png` | 9种方法性能对比 (6个子图) |
| `pregenant_roc_curves.png` | ROC 曲线对比 |
| `pregenant_shap_beeswarm.png` | SHAP beeswarm 图 |
| `pregenant_shap_bar.png` | Top 20 代谢物重要性 |
| `pregenant_shap_dependence.png` | Top 2 代谢物依赖图 |
| `pregenant_metabolite_distribution.png` | Top 10 代谢物分布箱线图 |
| `pregenant_correlation_heatmap.png` | Top 20 代谢物相关性热力图 |
| `pregenant_cv_results.png` | 交叉验证结果 |

## 🎯 关键特性

### 1. TabPFN 高亮显示
- 所有对比图表中 TabPFN 用蓝色高亮显示
- 便于快速识别 TabPFN 性能

### 2. 代谢组学专用处理
- 对数转换处理代谢物浓度数据
- 缺失值中位数填充
- 127维特征全面分析

### 3. SHAP 深度解释
- KernelExplainer 计算 SHAP 值
- Top 20 重要代谢物可视化
- 代谢物间相互作用分析

## 📊 预期结果示例

### Benchmark 排名 (示例)
```
⭐ TabPFN: 0.7500
   XGBoost: 0.7167
   LightGBM: 0.6833
   RandomForest: 0.6667
   ...
```

### Top 5 代谢物 (示例)
```
1. glucose: 0.2341
2. lactate: 0.1987
3. citrulline: 0.1765
4. glutamine: 0.1543
5. alanine: 0.1321
```

## ⚙️ 参数调整

### 修改 TabPFN 参数
```python
# 在 notebook 中找到以下代码并修改
TabPFNClassifier.create_default_for_version(
    ModelVersion.V2,
    n_estimators=4,  # 增加以提高性能
    device='cpu'     # 改为 'cuda' 使用 GPU
)
```

### 修改测试集比例
```python
# 默认 30% 测试集
train_test_split(X, y, test_size=0.3, ...)
```

### 修改 SHAP 样本数
```python
# 增加 nsamples 提高精度 (但速度更慢)
shap_values = explainer.shap_values(test_sample, nsamples=100)
```

## 🔧 依赖要求

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap tabpfn
```

## ⚠️ 注意事项

1. **运行时间**: 
   - Benchmark 对比: ~2-3 分钟
   - SHAP 计算: ~1-2 分钟
   - 总运行时间: ~5 分钟

2. **内存使用**: 
   - 峰值内存: ~2GB
   - 建议内存: 4GB+

3. **SHAP 计算**: 
   - 样本量较小时 (n=12 测试集)，SHAP 结果可能波动
   - 建议 nsamples=50 平衡速度和精度

## 📚 参考

- [TabPFN Documentation](https://github.com/PriorLabs/TabPFN)
- [SHAP Documentation](https://shap.readthedocs.io/)
- 原始数据: `/xuzhuo/LGLC/TabPFN/dataset/preganent raw data 202603(1).csv`
