#!/usr/bin/env python
"""Convert markdown to PDF using ReportLab with Chinese font support"""

import re
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Register Chinese fonts
pdfmetrics.registerFont(TTFont('NotoSans', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
pdfmetrics.registerFont(TTFont('NotoSansBold', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'))

# Configuration
md_file = 'ANALYSIS_REPORT.md'
output_pdf = 'Diabetes_Analysis_Report.pdf'

# Read markdown
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Create PDF
pdf = SimpleDocTemplate(
    output_pdf,
    pagesize=A4,
    rightMargin=2*cm,
    leftMargin=2*cm,
    topMargin=2*cm,
    bottomMargin=2*cm
)

# Define styles
styles = getSampleStyleSheet()

# Title style
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontName='NotoSansBold',
    fontSize=20,
    textColor=colors.HexColor('#1a237e'),
    spaceAfter=30,
    alignment=TA_CENTER,
    borderColor=colors.HexColor('#3498db'),
    borderWidth=2,
    borderPadding=10
)

# Heading 1 style (##)
h1_style = ParagraphStyle(
    'CustomH1',
    parent=styles['Heading2'],
    fontName='NotoSansBold',
    fontSize=16,
    textColor=colors.HexColor('#283593'),
    spaceAfter=12,
    spaceBefore=20,
    borderColor=colors.HexColor('#9fa8da'),
    borderWidth=1,
    borderPadding=5
)

# Heading 2 style (###)
h2_style = ParagraphStyle(
    'CustomH2',
    parent=styles['Heading3'],
    fontName='NotoSansBold',
    fontSize=13,
    textColor=colors.HexColor('#3949ab'),
    spaceAfter=10,
    spaceBefore=15
)

# Heading 3 style (####)
h3_style = ParagraphStyle(
    'CustomH3',
    parent=styles['Heading4'],
    fontName='NotoSansBold',
    fontSize=11,
    textColor=colors.HexColor('#5c6bc0'),
    spaceAfter=8,
    spaceBefore=10
)

# Body text style
body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['BodyText'],
    fontName='NotoSans',
    fontSize=10.5,
    leading=16,
    alignment=TA_JUSTIFY,
    spaceAfter=8
)

# Bullet style
bullet_style = ParagraphStyle(
    'CustomBullet',
    parent=styles['BodyText'],
    fontName='NotoSans',
    fontSize=10.5,
    leading=16,
    leftIndent=20,
    spaceAfter=5
)

# Caption style
caption_style = ParagraphStyle(
    'Caption',
    parent=styles['Italic'],
    fontName='NotoSans',
    fontSize=9,
    textColor=colors.gray,
    alignment=TA_CENTER,
    spaceAfter=10
)

# Process content
story = []
lines = md_content.split('\n')
i = 0

while i < len(lines):
    line = lines[i].strip()
    
    # Skip ---
    if line == '---':
        i += 1
        continue
    
    # Title
    if line.startswith('# ') and i == 0:
        title = line.replace('# ', '')
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))
    
    # Heading 1 (##)
    elif line.startswith('## '):
        title = line.replace('## ', '')
        story.append(Paragraph(title, h1_style))
    
    # Heading 2 (###)
    elif line.startswith('### '):
        title = line.replace('### ', '')
        story.append(Paragraph(title, h2_style))
    
    # Heading 3 (####)
    elif line.startswith('#### '):
        title = line.replace('#### ', '')
        story.append(Paragraph(title, h3_style))
    
    # Images
    elif line.startswith('![') and '](' in line:
        match = re.search(r'\]\(([^)]+)\)', line)
        if match:
            img_path = '../results/' + match.group(1)
            if os.path.exists(img_path):
                try:
                    img = Image(img_path, width=16*cm, height=None)
                    story.append(Spacer(1, 10))
                    story.append(img)
                    story.append(Spacer(1, 5))
                except:
                    pass
    
    # Table header
    elif line.startswith('|') and i + 1 < len(lines) and '---' in lines[i + 1]:
        # Collect table rows
        table_data = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            row = lines[i].strip()
            if '---' not in row:
                cells = [c.strip() for c in row.split('|')[1:-1]]
                table_data.append(cells)
            i += 1
        
        if table_data:
            # Create table
            table = Table(table_data, colWidths=[6*cm, 4*cm, 4*cm, 4*cm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5f5f5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'NotoSansBold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTNAME', (0, 1), (-1, -1), 'NotoSans'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(Spacer(1, 10))
            story.append(table)
            story.append(Spacer(1, 10))
        continue
    
    # Skip table separator
    elif line.startswith('|') and '---' in line:
        pass
    
    # Bullet points
    elif line.startswith('- ') or line.startswith('* '):
        text = line[2:]
        # Handle bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        story.append(Paragraph('• ' + text, bullet_style))
    
    # Empty line
    elif not line:
        story.append(Spacer(1, 5))
    
    # Regular paragraph
    else:
        # Handle bold text
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
        story.append(Paragraph(text, body_style))
    
    i += 1

# Build PDF
pdf.build(story)

print(f"✓ PDF 已生成: {output_pdf}")
print(f"  文件大小: {os.path.getsize(output_pdf) / 1024:.1f} KB")
