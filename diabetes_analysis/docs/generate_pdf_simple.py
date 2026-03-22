#!/usr/bin/env python
"""Generate PDF with images using PyMuPDF"""

import fitz
from pathlib import Path

# Create new PDF
doc = fitz.open()

# Page dimensions (A4 in points)
pw, ph = 595, 842
margin = 50

# Images to include
images = [
    ("../results/diabetes_benchmark_comparison.png", "图1: Benchmark 对比图"),
    ("../results/diabetes_prediction_scatter.png", "图2: 预测散点图"),
    ("../results/diabetes_residual_plots.png", "图3: 残差图"),
    ("../results/diabetes_shap_bar.png", "图4: SHAP 特征重要性"),
    ("../results/diabetes_shap_summary.png", "图5: SHAP Summary 图"),
    ("../results/diabetes_shap_dependence.png", "图6: SHAP 依赖关系"),
    ("../results/diabetes_correlation_heatmap.png", "图7: 特征相关性热力图"),
]

# Create cover page
page = doc.new_page(width=pw, height=ph)
# Title
page.insert_text((pw/2 - 150, 100), "Diabetes 数据集回归分析实验报告", fontsize=18, color=(0.1, 0.14, 0.49))
# Summary
summary_text = """摘要

本研究对 Diabetes 数据集进行了系统的回归分析，对比了 TabPFN 与其他 9 种经典机器学习回归模型的
性能。Diabetes 数据集包含 442 名糖尿病患者的 10 项生理指标。

实验结果表明，TabPFN 在该回归任务上取得了最佳性能(R²=0.507)，显著优于传统机器学习方法。

基于 SHAP 解释性分析，本研究识别出 BMI、s5(对数血清甘油三酯水平)和血压是影响糖尿病进展的
关键生理指标。"""

page.insert_textbox(fitz.Rect(margin, 150, pw-margin, 400), summary_text, fontsize=11)

# Results table
results_text = """主要结果:

模型排名 (按 R²):
1. TabPFN           R²=0.507  RMSE=51.60
2. LinearRegression R²=0.477  RMSE=53.12  
3. RandomForest     R²=0.470  RMSE=53.48
4. GradientBoosting R²=0.431  RMSE=55.42
5. Ridge            R²=0.423  RMSE=55.79"""

page.insert_textbox(fitz.Rect(margin, 420, pw-margin, 600), results_text, fontsize=10, color=(0.2, 0.2, 0.2))

# Add image pages
for img_path, caption in images:
    if Path(img_path).exists():
        page = doc.new_page(width=pw, height=ph)
        
        # Caption at top
        page.insert_text((margin, margin), caption, fontsize=12, color=(0.1, 0.14, 0.49))
        
        # Image below caption
        img_rect = fitz.Rect(margin, margin + 25, pw - margin, ph - margin)
        page.insert_image(img_rect, filename=img_path)

# Add results page
page = doc.new_page(width=pw, height=ph)
page.insert_text((margin, margin), "详细结果数据", fontsize=14, color=(0.1, 0.14, 0.49))

# Read and display CSV results
import pandas as pd
df = pd.read_csv('../results/diabetes_benchmark_results.csv')
df_sorted = df.sort_values('R2', ascending=False)

table_text = "模型性能对比:\n\n"
table_text += f"{'Model':<20} {'R2':>8} {'RMSE':>10} {'MAE':>10}\n"
table_text += "-" * 50 + "\n"

for _, row in df_sorted.iterrows():
    table_text += f"{row['Model']:<20} {row['R2']:>8.4f} {row['RMSE']:>10.2f} {row['MAE']:>10.2f}\n"

page.insert_textbox(fitz.Rect(margin, margin + 40, pw - margin, ph - margin), table_text, fontsize=9, fontname="courier")

# Save
output = "Diabetes_Analysis_Report_Final.pdf"
doc.save(output)
doc.close()

print(f"✓ PDF 已生成: {output}")
print(f"  文件大小: {Path(output).stat().st_size / 1024:.1f} KB")
print(f"  总页数: {len(fitz.open(output))}")
