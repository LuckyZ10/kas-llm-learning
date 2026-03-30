#!/usr/bin/env python3
"""
专业PDF学术书籍翻译工具 V2.0
- 保留原书排版结构
- 图片提取并保留位置
- 中英双语对照（术语保留英文）
- 高质量HTML+CSS重建
"""

import fitz  # PyMuPDF
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from PIL import Image
import io

@dataclass
class PDFElement:
    """PDF元素基类"""
    type: str  # 'text', 'image', 'math', 'table'
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)
    page_num: int
    content: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TextElement(PDFElement):
    """文本元素"""
    font_name: str = ""
    font_size: float = 12.0
    is_bold: bool = False
    is_italic: bool = False
    color: str = "#000000"
    
@dataclass
class ImageElement(PDFElement):
    """图片元素"""
    image_path: str = ""
    width: int = 0
    height: int = 0
    caption: str = ""

class ProfessionalPDFProcessor:
    """专业PDF处理器"""
    
    # 学术术语对照表（保留英文，中文注释）
    ACADEMIC_TERMS = {
        # 核心概念 - 保留英文，加中文注释
        "epitaxy": "epitaxy（外延）",
        "epitaxial": "epitaxial（外延的）",
        "epitaxial growth": "epitaxial growth（外延生长）",
        "semiconductor": "semiconductor（半导体）",
        "heterostructure": "heterostructure（异质结构）",
        "heterostructures": "heterostructures（异质结构）",
        
        # 材料
        "substrate": "substrate（衬底）",
        "layer": "layer（层）",
        "crystalline": "crystalline（晶体的）",
        "amorphous": "amorphous（非晶的）",
        "zincblende": "zincblende（闪锌矿）",
        "wurtzite": "wurtzite（纤锌矿）",
        
        # 物理概念
        "band gap": "band gap（带隙）",
        "band alignment": "band alignment（能带对齐）",
        "strain": "strain（应变）",
        "relaxation": "relaxation（弛豫）",
        "dislocation": "dislocation（位错）",
        "nucleation": "nucleation（成核）",
        
        # 生长方法 - 保留缩写
        "metalorganic vapor-phase epitaxy": "MOVPE（metalorganic vapor-phase epitaxy，金属有机气相外延）",
        "molecular-beam epitaxy": "MBE（molecular-beam epitaxy，分子束外延）",
        "liquid-phase epitaxy": "LPE（liquid-phase epitaxy，液相外延）",
        "MOVPE": "MOVPE（金属有机气相外延）",
        "MBE": "MBE（分子束外延）",
        "LPE": "LPE（液相外延）",
        
        # 纳米结构
        "quantum dot": "quantum dot（量子点）",
        "quantum dots": "quantum dots（量子点）",
        "quantum wire": "quantum wire（量子线）",
        "quantum wires": "quantum wires（量子线）",
        "nanowire": "nanowire（纳米线）",
        "nanowires": "nanowires（纳米线）",
        
        # 工艺概念
        "doping": "doping（掺杂）",
        "diffusion": "diffusion（扩散）",
        "annealing": "annealing（退火）",
        
        # 理论
        "thermodynamics": "thermodynamics（热力学）",
        "kinetics": "kinetics（动力学）",
        
        # 章节标题翻译
        "Preface": "Preface（前言）",
        "Contents": "Contents（目录）",
        "Chapter": "Chapter（章）",
        "References": "References（参考文献）",
        "Index": "Index（索引）",
    }
    
    # 保留不翻译的模式
    SKIP_PATTERNS = [
        r'^\d+$',  # 纯数字（页码）
        r'^[\dvxivlcmd\s\.]+$',  # 罗马数字页码
        r'^Fig\.?\s*\d+',  # 图注
        r'^Table\.?\s*\d+',  # 表注
        r'^\d+\.\d+',  # 章节编号
        r'https?://',  # URL
        r'^[A-Z][a-z]+\s+[A-Z][a-z]+$',  # 人名
        r'ISBN|ISSN|DOI|©',  # 出版标识
        r'^[A-Z][a-z]+\s+(University|Institute|College|Laboratory)',  # 机构名
        r'Springer',  # 出版社
        r'^[A-Z]{2,}$',  # 全大写缩写
    ]
    
    def __init__(self, pdf_path: str, output_dir: str = None):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        
        if output_dir is None:
            output_dir = self.pdf_path.stem + "_processed"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 图片输出目录
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
    def should_translate(self, text: str) -> bool:
        """判断是否应该翻译这段文本"""
        text = text.strip()
        if not text or len(text) < 2:
            return False
            
        for pattern in self.SKIP_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        return True
    
    def translate_with_terms(self, text: str) -> str:
        """翻译文本，保留学术术语中英对照"""
        if not self.should_translate(text):
            return text
        
        translated = text
        
        # 1. 首先处理学术术语（按长度降序，避免部分替换）
        for en, zh_en in sorted(self.ACADEMIC_TERMS.items(), key=lambda x: -len(x[0])):
            # 整词匹配，忽略大小写
            pattern = r'\b' + re.escape(en) + r'\b'
            translated = re.sub(pattern, zh_en, translated, flags=re.IGNORECASE)
        
        # 2. 检测数学公式并标记
        math_patterns = [
            (r'([α-ωΑ-Ω])', r'\1'),  # 希腊字母
            (r'(∫|∑|∏|√|∞|±|×|÷|∂|∇)', r'\1'),  # 数学符号
            (r'(_\{[^}]+\})', r'\1'),  # 下标
            (r'(\^\{[^}]+\})', r'\1'),  # 上标
            (r'(\\[a-zA-Z]+)', r'\1'),  # LaTeX命令
        ]
        
        # 3. 简单的整句翻译提示（实际可以接入翻译API）
        # 这里我们主要是做术语替换，完整的句子翻译需要更好的策略
        
        return translated
    
    def extract_images_from_page(self, page_num: int) -> List[ImageElement]:
        """从页面提取图片"""
        page = self.doc[page_num]
        image_list = page.get_images()
        images = []
        
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = self.doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # 保存图片
            image_filename = f"page_{page_num + 1}_img_{img_index}.{image_ext}"
            image_path = self.images_dir / image_filename
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            # 获取图片在页面上的位置
            rects = page.get_image_rects(xref)
            bbox = rects[0] if rects else (0, 0, 100, 100)
            if hasattr(bbox, '__iter__'):
                bbox = tuple(float(x) for x in bbox)
            
            # 获取图片尺寸
            pil_img = Image.open(io.BytesIO(image_bytes))
            width, height = pil_img.size
            
            images.append(ImageElement(
                type='image',
                bbox=bbox,
                page_num=page_num + 1,
                image_path=str(image_path.relative_to(self.output_dir)),
                width=width,
                height=height
            ))
        
        return images
    
    def extract_text_with_style(self, page_num: int) -> List[TextElement]:
        """提取带样式的文本"""
        page = self.doc[page_num]
        text_elements = []
        
        # 使用dict模式获取详细格式
        text_dict = page.get_text("dict")
        
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                line_text = ""
                line_font_size = 12
                line_font_name = ""
                line_is_bold = False
                line_is_italic = False
                line_color = "#000000"
                
                for span in line["spans"]:
                    line_text += span["text"]
                    line_font_size = span.get("size", 12)
                    line_font_name = span.get("font", "")
                    flags = span.get("flags", 0)
                    line_is_bold = bool(flags & 2**4)  #  bold flag
                    line_is_italic = bool(flags & 2**0)  # italic flag
                    
                    # 获取颜色
                    color = span.get("color", 0)
                    line_color = f"#{color:06x}"
                
                if line_text.strip():
                    # 转换bbox为tuple
                    bbox = line["bbox"]
                    if hasattr(bbox, '__iter__'):
                        bbox = tuple(float(x) for x in bbox)
                    
                    text_elements.append(TextElement(
                        type='text',
                        bbox=bbox,
                        page_num=page_num + 1,
                        content=line_text,
                        font_name=line_font_name,
                        font_size=line_font_size,
                        is_bold=line_is_bold,
                        is_italic=line_is_italic,
                        color=line_color
                    ))
        
        return text_elements
    
    def analyze_page_structure(self, page_num: int) -> Dict:
        """分析页面结构"""
        page = self.doc[page_num]
        page_rect = page.rect
        
        # 转换page_rect为普通数值
        page_width = float(page_rect.width)
        page_height = float(page_rect.height)
        
        # 提取所有元素
        text_elements = self.extract_text_with_style(page_num)
        images = self.extract_images_from_page(page_num)
        
        # 分析布局区域
        header_zone = page_height * 0.1  # 顶部10%
        footer_zone = page_height * 0.9  # 底部10%
        margin_left = page_width * 0.1
        margin_right = page_width * 0.9
        
        # 分类元素
        header_elements = []
        footer_elements = []
        body_elements = []
        
        for elem in text_elements:
            y_center = (elem.bbox[1] + elem.bbox[3]) / 2
            
            if y_center < header_zone:
                header_elements.append(elem)
            elif y_center > footer_zone:
                footer_elements.append(elem)
            else:
                body_elements.append(elem)
        
        # 检测标题（大字体、粗体）
        titles = [e for e in body_elements if e.font_size > 14 or e.is_bold]
        
        def elem_to_dict(e):
            """安全地将元素转换为字典"""
            d = {}
            for key, value in e.__dict__.items():
                if key == 'bbox' and value is not None:
                    # 特别处理bbox，确保是tuple of floats
                    try:
                        d[key] = tuple(float(x) for x in value)
                    except:
                        d[key] = (0, 0, 0, 0)
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    d[key] = list(value) if not isinstance(value, tuple) else value
                else:
                    d[key] = value
            return d
        
        return {
            'page_num': page_num + 1,
            'width': page_width,
            'height': page_height,
            'header': [elem_to_dict(e) for e in header_elements],
            'footer': [elem_to_dict(e) for e in footer_elements],
            'titles': [elem_to_dict(e) for e in titles],
            'body': [elem_to_dict(e) for e in body_elements if e not in titles],
            'images': [elem_to_dict(e) for e in images]
        }
    
    def process_pages(self, start_page: int = 0, end_page: int = None):
        """处理指定页面范围"""
        if end_page is None:
            end_page = self.total_pages
        
        all_pages = []
        
        print(f"处理页面 {start_page + 1} 到 {end_page}...")
        
        for page_num in range(start_page, end_page):
            if (page_num - start_page) % 10 == 0:
                print(f"  进度: {page_num - start_page + 1}/{end_page - start_page}")
            
            page_data = self.analyze_page_structure(page_num)
            
            # 翻译文本内容
            for elem_list in ['header', 'footer', 'titles', 'body']:
                for elem in page_data[elem_list]:
                    elem['translated'] = self.translate_with_terms(elem['content'])
            
            all_pages.append(page_data)
        
        # 保存结构化数据
        output_json = self.output_dir / "processed_data.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump({
                'source': str(self.pdf_path),
                'total_pages': self.total_pages,
                'processed_pages': len(all_pages),
                'pages': all_pages
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 数据处理完成，保存到: {output_json}")
        return all_pages
    
    def generate_professional_html(self, pages_data: List[Dict], output_html: str):
        """生成专业排版的HTML"""
        
        html_template = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>半导体外延：异质结构的物理与制备（中英对照版）</title>
    <style>
        @page {
            size: A4;
            margin: 2cm 1.8cm 2.5cm 1.8cm;
            @top-center {
                content: "半导体外延：异质结构的物理与制备";
                font-size: 9pt;
                color: #666;
            }
            @bottom-center {
                content: counter(page);
                font-size: 10pt;
            }
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: "Noto Serif CJK SC", "Source Han Serif SC", "SimSun", serif;
            font-size: 11pt;
            line-height: 1.8;
            color: #333;
            text-align: justify;
        }
        
        /* 封面页 */
        .cover {
            page-break-after: always;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            text-align: center;
        }
        
        .cover-title {
            font-size: 28pt;
            font-weight: bold;
            color: #1a1a1a;
            margin: 30pt 0;
            line-height: 1.4;
        }
        
        .cover-subtitle {
            font-size: 16pt;
            color: #444;
            margin: 15pt 0;
        }
        
        .cover-author {
            font-size: 14pt;
            color: #555;
            margin: 20pt 0;
        }
        
        .cover-edition {
            font-size: 12pt;
            color: #666;
            margin: 15pt 0;
        }
        
        /* 正文页 */
        .page {
            page-break-before: always;
            position: relative;
            min-height: 100%;
        }
        
        .page:first-of-type {
            page-break-before: auto;
        }
        
        /* 页眉 */
        .page-header {
            text-align: center;
            font-size: 9pt;
            color: #666;
            border-bottom: 0.5pt solid #ccc;
            padding-bottom: 8pt;
            margin-bottom: 15pt;
        }
        
        /* 页脚 */
        .page-footer {
            text-align: center;
            font-size: 9pt;
            color: #666;
            border-top: 0.5pt solid #ccc;
            padding-top: 8pt;
            margin-top: 20pt;
        }
        
        /* 标题层级 */
        h1 {
            font-size: 20pt;
            font-weight: bold;
            color: #1a1a1a;
            margin: 25pt 0 15pt 0;
            text-align: center;
            page-break-after: avoid;
        }
        
        h2 {
            font-size: 16pt;
            font-weight: bold;
            color: #2c3e50;
            margin: 20pt 0 12pt 0;
            page-break-after: avoid;
        }
        
        h3 {
            font-size: 13pt;
            font-weight: bold;
            color: #34495e;
            margin: 15pt 0 10pt 0;
            page-break-after: avoid;
        }
        
        /* 段落 */
        p {
            margin: 10pt 0;
            text-indent: 2em;
        }
        
        p.no-indent {
            text-indent: 0;
        }
        
        /* 图片 */
        .figure {
            margin: 20pt 0;
            text-align: center;
            page-break-inside: avoid;
        }
        
        .figure img {
            max-width: 100%;
            height: auto;
        }
        
        .figure-caption {
            font-size: 10pt;
            color: #555;
            margin-top: 8pt;
            text-align: center;
        }
        
        /* 中英对照样式 */
        .term-en {
            font-style: italic;
            color: #2c3e50;
        }
        
        .term-zh {
            color: #666;
        }
        
        /* 数学公式 */
        .math {
            font-family: "Times New Roman", "Latin Modern Math", serif;
            font-style: italic;
            text-align: center;
            margin: 15pt 0;
            padding: 10pt;
        }
        
        /* 代码 */
        code {
            font-family: "Fira Code", "Consolas", monospace;
            font-size: 9.5pt;
            background-color: #f4f4f4;
            padding: 2pt 4pt;
            border-radius: 2pt;
        }
        
        pre {
            font-family: "Fira Code", "Consolas", monospace;
            font-size: 9pt;
            background-color: #f8f8f8;
            border: 1pt solid #e0e0e0;
            padding: 12pt;
            margin: 12pt 0;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        /* 特殊块 */
        .abstract {
            background-color: #f9f9f9;
            border-left: 3pt solid #3498db;
            padding: 12pt 15pt;
            margin: 15pt 0;
        }
        
        .abstract-title {
            font-weight: bold;
            margin-bottom: 8pt;
        }
        
        /* 版权信息 */
        .copyright {
            font-size: 9pt;
            color: #666;
            margin: 30pt 0;
            padding: 15pt;
            border: 1pt solid #ddd;
        }
        
        /* 打印优化 */
        @media print {
            body {
                font-size: 10.5pt;
            }
            
            .figure {
                page-break-inside: avoid;
            }
            
            pre, blockquote {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>
'''
        
        html_content = html_template
        
        for page_data in pages_data:
            page_num = page_data['page_num']
            
            html_content += f'\n<!-- Page {page_num} -->\n'
            html_content += f'<div class="page" data-page="{page_num}">\n'
            
            # 页眉
            if page_data['header']:
                header_text = ' '.join([h.get('translated', h['content']) for h in page_data['header']])
                html_content += f'  <div class="page-header">{self._escape_html(header_text)}</div>\n'
            
            # 图片
            for img in page_data['images']:
                html_content += f'  <div class="figure">\n'
                html_content += f'    <img src="{img['image_path']}" alt="Figure" />\n'
                if img.get('caption'):
                    html_content += f'    <div class="figure-caption">{self._escape_html(img['caption'])}</div>\n'
                html_content += f'  </div>\n'
            
            # 标题
            for title in page_data['titles']:
                level = self._detect_heading_level(title)
                content = title.get('translated', title['content'])
                html_content += f'  <h{level}>{self._escape_html(content)}</h{level}>\n'
            
            # 正文
            for para in page_data['body']:
                content = para.get('translated', para['content'])
                if content.strip():
                    # 检测是否是特殊段落（版权、摘要等）
                    if 'copyright' in content.lower() or '©' in content:
                        html_content += f'  <div class="copyright">{self._escape_html(content)}</div>\n'
                    else:
                        html_content += f'  <p>{self._escape_html(content)}</p>\n'
            
            # 页脚
            if page_data['footer']:
                footer_text = ' '.join([f.get('translated', f['content']) for f in page_data['footer']])
                html_content += f'  <div class="page-footer">{self._escape_html(footer_text)}</div>\n'
            
            html_content += '</div>\n'
        
        html_content += '</body>\n</html>'
        
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML生成完成: {output_html}")
        return output_html
    
    def _escape_html(self, text: str) -> str:
        """转义HTML特殊字符"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))
    
    def _detect_heading_level(self, title_elem: Dict) -> int:
        """检测标题级别"""
        font_size = title_elem.get('font_size', 12)
        is_bold = title_elem.get('is_bold', False)
        
        if font_size >= 18 or (font_size >= 14 and is_bold):
            return 1
        elif font_size >= 14 or is_bold:
            return 2
        else:
            return 3

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python pdf_professional_processor.py <pdf文件> [起始页] [结束页]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    start_page = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0
    end_page = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    processor = ProfessionalPDFProcessor(pdf_path)
    pages_data = processor.process_pages(start_page, end_page)
    
    output_html = processor.output_dir / "translated_book.html"
    processor.generate_professional_html(pages_data, output_html)
    
    print(f"\n📁 输出目录: {processor.output_dir}")
    print(f"📄 HTML文件: {output_html}")
    print(f"🖼️ 图片目录: {processor.images_dir}")
    print(f"\n下一步: 使用Playwright或wkhtmltopdf转换为PDF")
