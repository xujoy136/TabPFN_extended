#!/usr/bin/env python
"""Convert markdown to styled PDF using fpdf2 with proper Chinese font support"""

import re
import os
from fpdf import FPDF

# Configuration
md_file = 'ANALYSIS_REPORT.md'
output_pdf = 'Diabetes_Analysis_Report_Fixed.pdf'

# Parse markdown
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Custom PDF class
class PDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=15)
        # Add Unicode font - use ttc files directly
        self.add_font('Noto', '', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', uni=True)
        self.add_font('Noto', 'B', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', uni=True)
        self.add_font('Noto', 'I', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', uni=True)
        self.set_font('Noto', '', 11)

    def footer(self):
        self.set_y(-10)
        self.set_font('Noto', 'I', 8)
        self.cell(0, 10, f'第 {self.page_no()} 页', align='C')

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font('Noto', 'B', 16)
            self.set_text_color(26, 35, 126)
            self.ln(8)
            self.cell(0, 10, title, ln=True, align='L')
            self.set_text_color(0, 0, 0)
            self.ln(2)
        elif level == 2:
            self.set_font('Noto', 'B', 13)
            self.set_text_color(40, 53, 147)
            self.ln(6)
            self.cell(0, 8, title, ln=True, align='L')
            self.set_text_color(0, 0, 0)
            self.ln(2)
        elif level == 3:
            self.set_font('Noto', 'B', 11)
            self.set_text_color(57, 73, 171)
            self.ln(4)
            self.cell(0, 7, title, ln=True, align='L')
            self.set_text_color(0, 0, 0)
            self.ln(1)

    def add_formatted_text(self, text):
        """Handle mixed text with proper encoding"""
        self.set_font('Noto', '', 10)
        self.set_text_color(51, 51, 51)
        
        # Handle bold markers
        parts = re.split(r'(\*\*[^*]+\*\*)', text)
        
        for part in parts:
            if not part:
                continue
            if part.startswith('**') and part.endswith('**'):
                self.set_font('Noto', 'B', 10)
                self.write(6, part[2:-2])
                self.set_font('Noto', '', 10)
            else:
                self.set_font('Noto', '', 10)
                self.write(6, part)
        
        self.ln(5)

    def add_bullet_point(self, text):
        """Add bullet point"""
        self.set_font('Noto', '', 10)
        self.cell(5, 6, '-', ln=0)
        self.multi_cell(0, 6, text)
        self.ln(1)

# Create PDF
pdf = PDF()
pdf.add_page()

# Title
pdf.set_font('Noto', 'B', 18)
pdf.set_text_color(26, 35, 126)
pdf.cell(0, 15, 'Diabetes 数据集回归分析实验报告', ln=True, align='C')
pdf.ln(5)

# Process content
lines = md_content.split('\n')
i = 0
while i < len(lines):
    line = lines[i].strip()
    
    # Skip title header
    if line.startswith('# Diabetes') or line.startswith('---'):
        i += 1
        continue
    
    # Headers
    if line.startswith('## '):
        title = line.replace('## ', '')
        pdf.chapter_title(title, level=1)
    elif line.startswith('### '):
        title = line.replace('### ', '')
        pdf.chapter_title(title, level=2)
    elif line.startswith('#### '):
        title = line.replace('#### ', '')
        pdf.chapter_title(title, level=3)
    
    # Images
    elif line.startswith('![') and '](' in line:
        match = re.search(r'\]\(([^)]+)\)', line)
        if match:
            img_path = '../results/' + match.group(1)
            if os.path.exists(img_path):
                try:
                    pdf.image(img_path, x=20, w=170)
                    pdf.ln(3)
                except:
                    pass
    
    # Table header separator
    elif line.startswith('| ---') or line.startswith('|:---'):
        pass  # Skip table separator
    
    # Table rows
    elif line.startswith('|') and '|' in line[1:]:
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if cells and not all(c.replace('-', '').replace(':', '') == '' for c in cells):
            pdf.set_font('Noto', '', 9)
            for cell in cells:
                pdf.cell(40, 6, cell, border=1)
            pdf.ln()
    
    # Bullet points
    elif line.startswith('- ') or line.startswith('* '):
        text = line[2:]
        pdf.add_bullet_point(text)
    
    # Empty line
    elif not line:
        pdf.ln(2)
    
    # Regular paragraph
    else:
        # Handle bold text
        if '**' in line:
            pdf.add_formatted_text(line)
        else:
            pdf.set_font('Noto', '', 10)
            pdf.multi_cell(0, 6, line)
            pdf.ln(2)
    
    i += 1

# Save PDF
pdf.output(output_pdf)
print(f"✓ PDF 已保存: {os.path.abspath(output_pdf)}")
print(f"  文件大小: {os.path.getsize(output_pdf) / 1024:.1f} KB")
