# SHAP Analysis for TabPFN Regression Model

使用 SHAP (SHapley Additive exPlanations) 对 TabPFN 回归模型进行可解释性分析的项目。

## 📁 项目结构

```
shap_analysis_tabpfn/
├── README.md                          # 本说明文档
├── requirements.txt                   # Python 依赖
├── data_loader.py                     # 数据加载工具模块
├── shap_analysis.py                   # 主分析脚本（基础版）
├── shap_analysis_v2.py                # 增强版脚本（支持本地数据）
├── diabetes_dataset.csv               # 📊 本地数据集（92.9 KB）
├── shap_feature_importance_bar.png   # 特征重要性条形图
├── shap_beeswarm.png                  # 蜂群图
└── shap_dependence_top2.png           # 依赖图
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd shap_analysis_tabpfn
pip install -r requirements.txt
```

### 2. 运行分析（三种方式）

#### 方式 A：使用 sklearn 内置数据（默认）
```bash
python shap_analysis.py
# 或
python shap_analysis_v2.py
```

#### 方式 B：使用本地 CSV 数据
```bash
python shap_analysis_v2.py --use-local
```

#### 方式 C：先保存再分析
```bash
# 保存 sklearn 数据到 CSV
python data_loader.py --save

# 然后从 CSV 分析
python shap_analysis_v2.py --use-local
```

## 📊 数据集

### 本地数据集：`diabetes_dataset.csv`

已包含在项目中，无需联网下载！

| 属性 | 数值 |
|------|------|
| **文件大小** | 92.9 KB |
| **样本数** | 442 |
| **特征数** | 10 |
| **目标变量** | 一年后疾病进展指标 |

### 特征说明

| 特征 | 说明 | 类型 |
|------|------|------|
| `age` | 年龄 | 数值 |
| `sex` | 性别 | 类别 (-0.044, 0.050) |
| `bmi` | 体重指数 | 数值 |
| `bp` | 平均血压 | 数值 |
| `s1` | 血清总胆固醇 | 数值 |
| `s2` | 低密度脂蛋白 | 数值 |
| `s3` | 高密度脂蛋白 | 数值 |
| `s4` | 总胆固醇/HDL 比值 | 数值 |
| `s5` | 血清甘油三酯的对数 | 数值 |
| `s6` | 血糖水平 | 数值 |
| `target` | 疾病进展指标 | 数值 (25-346) |

### 数据查看

```bash
# 检查本地数据集信息
python data_loader.py --check

# 输出示例：
# ============================================================
# Local Dataset Found
# ============================================================
# File: /path/to/diabetes_dataset.csv
# Size: 92.9 KB
#
# Samples: 442
# Features: 10
# Target statistics:
#   Range: [25.0, 346.0]
#   Mean: 152.13
#   Std: 77.09
```

## 🔍 数据加载模块 (`data_loader.py`)

提供灵活的数据加载功能：

```python
from data_loader import load_from_sklearn, load_from_csv, save_dataset

# 方式1: 从 sklearn 加载（无需本地文件）
X_train, X_test, y_train, y_test, feature_names = load_from_sklearn()

# 方式2: 从本地 CSV 加载（更快，不依赖 sklearn）
X_train, X_test, y_train, y_test, feature_names = load_from_csv('diabetes_dataset.csv')

# 方式3: 保存 sklearn 数据到 CSV
save_dataset('diabetes_dataset.csv')
```

### data_loader.py 命令行用法

```bash
# 保存数据集到 CSV
python data_loader.py --save

# 指定保存路径
python data_loader.py --save --filepath data/my_diabetes.csv

# 检查本地数据集
python data_loader.py --check

# 检查指定路径的数据集
python data_loader.py --check --filepath data/my_diabetes.csv
```

## 🛠️ 脚本对比

| 脚本 | 特点 | 推荐使用场景 |
|------|------|-------------|
| `shap_analysis.py` | 基础版本，从 sklearn 加载 | 快速开始，简单分析 |
| `shap_analysis_v2.py` | 增强版本，支持本地 CSV | 需要重复运行、自定义数据 |

### shap_analysis_v2.py 命令行参数

```bash
# 基础用法（从 sklearn 加载）
python shap_analysis_v2.py

# 使用本地 CSV
python shap_analysis_v2.py --use-local

# 使用更多 ensemble（更准但更慢）
python shap_analysis_v2.py --n-estimators 4

# 指定输出目录
python shap_analysis_v2.py --output-dir results/

# 保存数据然后分析
python shap_analysis_v2.py --save-data --use-local

# 组合使用
python shap_analysis_v2.py --use-local --n-estimators 4 --output-dir results/
```

## 📈 主要发现

### 特征重要性排名

| 排名 | 特征 | 平均 |SHAP| 解释 |
|:---:|:-----|:-----------:|------|
| 1️⃣ | **bmi** | 24.38 | **最重要** - 体重指数 |
| 2️⃣ | **s5** | 22.15 | 血清甘油三酯 |
| 3️⃣ | **bp** | 12.17 | 血压 |
| 4️⃣ | **sex** | 10.85 | 性别 |
| 5️⃣ | **s1** | 8.91 | 总胆固醇 |

### 关键洞察

1. **BMI 是最重要的预测因子** - 符合医学常识：肥胖是糖尿病的主要风险因素
2. **S5（血脂指标）排第二** - 血脂异常与糖尿病密切相关
3. **血压排第三** - 高血压常与糖尿病共存（代谢综合征）

## 📊 可视化说明

### 生成的图表

1. **shap_feature_importance_bar.png** - 特征重要性排名
2. **shap_beeswarm.png** - 全局特征影响分布
3. **shap_dependence_top2.png** - 前两个特征的依赖关系

### 图表解读

#### 蜂群图 (Beeswarm)
- **颜色**: 特征值高低（蓝=低，红=高）
- **X轴**: SHAP 值（右=正向影响，左=负向影响）
- **观察**: 高 BMI（红色）→ 推高疾病进展指标

#### 依赖图 (Dependence)
- 展示特征值与 SHAP 值的单调关系
- BMI 几乎呈完美线性：BMI 越高 → 影响越大

## 💡 使用自己的数据

如果你想分析自己的数据集：

```python
from data_loader import load_from_csv
from shap_analysis_v2 import train_and_evaluate, compute_shap_values, analyze_and_visualize

# 1. 准备你的 CSV 文件（必须有 target 列）
# 格式: feature1, feature2, ..., target

# 2. 加载数据
X_train, X_test, y_train, y_test, feature_names = load_from_csv(
    'your_data.csv',
    target_column='target'  # 修改为你的目标列名
)

# 3. 运行分析
model = train_and_evaluate(X_train, X_test, y_train, y_test)
shap_values, explainer, X_test_sample = compute_shap_values(model, X_train, X_test)
analyze_and_visualize(shap_values, X_test_sample, feature_names)
```

## 🎛️ 高级配置

### 修改集成数量

```bash
# 更高的 n_estimators = 更好的精度但更慢
python shap_analysis_v2.py --n-estimators 8
```

| n_estimators | 速度 | 精度 | 推荐场景 |
|-------------|:----:|:----:|----------|
| 1 | ⚡ 快 | 良好 | 快速实验 |
| 4 | 🐢 中等 | 更好 | 默认推荐 |
| 8 | 🐌 慢 | 最佳 | 最终分析 |

### 增加 SHAP 样本数

编辑脚本中的参数：

```python
shap_values, explainer, X_test_sample = compute_shap_values(
    model, X_train, X_test,
    n_background=100,    # 背景样本（默认50）
    n_test_samples=100   # 测试样本（默认50）
)
```

## 📝 技术说明

### 为什么使用 KernelExplainer？

TabPFN 是基于 Transformer 的模型：
- ❌ 不是树模型 → 不能用 TreeExplainer
- ✅ 通用模型 → 使用 KernelExplainer（模型无关）

### 计算时间参考

| 配置 | 时间 | 说明 |
|------|------|------|
| n_test=50, n_estimators=1 | ~30秒 | 快速预览 |
| n_test=100, n_estimators=4 | ~2分钟 | 标准分析 |
| n_test=200, n_estimators=8 | ~10分钟 | 深度分析 |

## 📚 参考资源

- **TabPFN 文档**: https://priorlabs.ai/docs
- **SHAP 文档**: https://shap.readthedocs.io/
- **糖尿病数据集**: https://scikit-learn.org/stable/datasets/toy_dataset.html
- **SHAP 论文**: Lundberg & Lee (2017)

## 🤝 贡献

欢迎改进：
1. 添加更多可视化类型
2. 支持分类任务
3. 优化计算性能
4. 添加更多数据集示例

## 📄 许可证

MIT License

---

**作者**: Generated by Claude Code  
**日期**: 2026-03-20  
**版本**: 2.0
