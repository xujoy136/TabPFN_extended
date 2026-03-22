#!/usr/bin/env python
"""Convert markdown to styled PDF using fpdf2 with Chinese font support"""

import re
import os
from fpdf import FPDF, XPos, YPos
from PIL import Image

# Configuration
md_file = "ANALYSIS_REPORT.md"
output_pdf = "Diabetes_Analysis_Report.pdf"
base_dir = '../results'

# Parse markdown
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Custom PDF class
class PDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=15)
        # Add Unicode font
        self.add_font('Noto', '', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        self.add_font('Noto', 'B', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
        self.add_font('Noto', 'I', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        self.set_font('Noto', '', 11)

    def header(self):
        pass

    def footer(self):
        self.set_y(-10)
        self.set_font('Noto', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title, level=1):
        if level == 1:
            self.set_font('Noto', 'B', 16)
            self.set_text_color(26, 35, 126)
            self.ln(8)
            self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.set_text_color(0, 0, 0)
            self.ln(2)
        elif level == 2:
            self.set_font('Noto', 'B', 14)
            self.set_text_color(40, 53, 147)
            self.ln(6)
            self.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.set_text_color(0, 0, 0)
            self.ln(2)
        elif level == 3:
            self.set_font('Noto', 'B', 12)
            self.set_text_color(57, 73, 171)
            self.ln(5)
            self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.set_text_color(0, 0, 0)
            self.ln(2)

    def add_formatted_paragraph(self, text):
        """Handle mixed Chinese/English text and bold formatting using write()"""
        self.set_text_color(51, 51, 51)

        # Split by bold markers **...**
        parts = re.split(r'(\*\*[^*]+\*\*)', text)

        for part in parts:
            if not part:
                continue

            if part.startswith('**') and part.endswith('**'):
                # Bold text
                self.set_font('Noto', 'B', 11)
                self.write(6, part[2:-2])
                self.set_font('Noto', '', 11)
            else:
                # Regular text - use write for better mixed text handling
                self.set_font('Noto', '', 11)
                # Add small space after English words for better wrapping
                text_to_write = part
                self.write(6, text_to_write)

        self.ln(6)

    def add_image(self, img_path, caption=''):
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return

        try:
            img = Image.open(img_path)
            img_w, img_h = img.size

            # A4 dimensions: 210 x 297 mm
            max_w = 170
            max_h = 120

            # Calculate scaling
            scale = min(max_w / img_w, max_h / img_h)
            new_w = img_w * scale
            new_h = img_h * scale

            # Center image
            x = (210 - new_w) / 2

            self.ln(5)
            self.image(img_path, x=x, w=new_w)
            self.ln(2)

            if caption:
                self.set_font('Noto', 'I', 9)
                self.set_text_color(102, 102, 102)
                self.cell(0, 5, caption, align='C')
                self.ln(3)
                self.set_text_color(0, 0, 0)
        except Exception as e:
            print(f"Warning: Could not add image {img_path}: {e}")

# Create PDF
pdf = PDF()
pdf.add_page()

# Title
pdf.set_font('Noto', 'B', 20)
pdf.set_text_color(26, 35, 126)
pdf.cell(0, 20, 'Diabetes 数据集回归分析实验报告', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
pdf.ln(10)

# Process content line by line
lines = md_content.split('\n')

for line in lines:
    line = line.strip()

    if line.startswith('# 结果'):
        continue

    elif line.startswith('## '):
        title = line.replace('## ', '')
        pdf.chapter_title(title, level=1)

    elif line.startswith('### '):
        title = line.replace('### ', '')
        pdf.chapter_title(title, level=2)

    elif line.startswith('**') and '**' in line[2:]:
        # Line starting with bold text
        pdf.ln(3)
        formatted_line = line.replace('**', '')
        pdf.add_formatted_paragraph(formatted_line)

    elif line.startswith('![alt text]'):
        # Image
        match = re.search(r'\(([^)]+)\)', line)
        if match:
            img_file = match.group(1)
            img_path = os.path.join(base_dir, img_file)
            pdf.add_image(img_path)

    elif line and len(line) > 0:
        # All other content - use formatted paragraph for better handling
        pdf.add_formatted_paragraph(line)

# Save PDF
pdf.output(output_pdf)

print(f"PDF saved to: {os.path.abspath(output_pdf)}")
