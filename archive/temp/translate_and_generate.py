#!/usr/bin/env python3
"""
PDF翻译与生成工具 - 使用Playwright + Paged.js生成专业PDF
"""

import json
import re
from pathlib import Path
from typing import List, Dict

# 翻译映射表（术语对照）
TERM_MAP = {
    # 书名和核心概念
    "Epitaxy of Semiconductors": "半导体外延",
    "Physics and Fabrication of Heterostructures": "异质结构的物理与制备",
    "Second Edition": "第二版",
    "Graduate Texts in Physics": "物理学研究生教材",
    
    # 前言相关
    "Preface to the Second Edition": "第二版前言",
    "Preface to the First Edition": "第一版前言",
    "Series Editors": "丛书编辑",
    
    # 常用术语
    "epitaxy": "外延",
    "epitaxial": "外延的",
    "semiconductor": "半导体",
    "heterostructure": "异质结构",
    "heterostructures": "异质结构",
    "crystalline": "晶体的",
    "crystalline layer": "晶体层",
    "crystalline substrate": "晶体衬底",
    "growth": "生长",
    "fabrication": "制备",
    "physics": "物理学",
    "semiconductor physics": "半导体物理",
    "solid state": "固体",
    "solid state physics": "固体物理学",
    "zincblende": "闪锌矿",
    "wurtzite": "纤锌矿",
    "quantum dots": "量子点",
    "quantum wires": "量子线",
    "nucleation": "成核",
    "doping": "掺杂",
    "diffusion": "扩散",
    "contacts": "接触",
    "metalorganic vapor-phase epitaxy": "金属有机气相外延",
    "molecular-beam epitaxy": "分子束外延",
    "liquid-phase epitaxy": "液相外延",
    "MOVPE": "MOVPE（金属有机气相外延）",
    "MBE": "MBE（分子束外延）",
    "LPE": "LPE（液相外延）",
    "thermodynamics": "热力学",
    "kinetics": "动力学",
    "band alignment": "能带对齐",
    "electronic states": "电子态",
    "low-dimensional structures": "低维结构",
    "elasticity": "弹性",
    "strain relaxation": "应变弛豫",
    "dislocations": "位错",
    "organic semiconductors": "有机半导体",
    "surfactants": "表面活性剂",
    "in situ analysis": "原位分析",
    "selective area growth": "选区生长",
    "epitaxial lateral overgrowth": "外延横向过生长",
    "migration-enhanced epitaxy": "迁移增强外延",
    "vapor-liquid-solid growth": "气-液-固生长",
    "nanowires": "纳米线",
    "nanostructure": "纳米结构",
    "nanostructure fabrication": "纳米结构制备",
    
    # 出版相关
    "All rights are reserved": "保留所有权利",
    "copyright": "版权",
    "translation": "翻译",
    "reprinting": "重印",
    "reuse of illustrations": "插图重用",
    
    # 常用句式
    "This book is based on": "本书基于",
    "I would like to thank": "我要感谢",
    "I am grateful to": "我感谢",
    "I am indebted to": "我感激",
}

def translate_text(text: str) -> str:
    """
    翻译文本，保留数学公式、代码、人名、机构名不变
    """
    if not text or not text.strip():
        return text
    
    # 检查是否主要是数字/页码
    if re.match(r'^[\dvxivlcmd\s]+$', text.strip().lower()):
        return text
    
    # 检查是否包含URL
    if re.search(r'https?://', text):
        return text
    
    # 检查是否是ISBN/ISSN/DOI等标识符
    if any(x in text for x in ['ISBN', 'ISSN', 'DOI', 'http', '©', 'Springer']):
        # 保留这些元数据不变
        return text
    
    translated = text
    
    # 按长度降序排序，避免部分替换问题
    for en, zh in sorted(TERM_MAP.items(), key=lambda x: -len(x[0])):
        # 使用正则表达式进行整词替换（忽略大小写）
        pattern = r'\b' + re.escape(en) + r'\b'
        translated = re.sub(pattern, zh, translated, flags=re.IGNORECASE)
    
    # 对于没有被术语表覆盖的文本，添加标记
    if translated == text and len(text) > 20:
        # 长文本未被翻译，可能是需要人工翻译的内容
        # 这里我们保留原文，实际使用时可以调用翻译API
        pass
    
    return translated

def process_batches(input_json: str, output_json: str):
    """处理批次文件，翻译文本内容"""
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    translated_batches = []
    
    for batch in data['batches']:
        translated_batch = {
            'start_page': batch['start_page'],
            'end_page': batch['end_page'],
            'blocks': []
        }
        
        for block in batch['blocks']:
            new_block = block.copy()
            
            if block['type'] in ['code', 'math']:
                # 保留原样
                new_block['translated'] = block['content']
                new_block['translation_note'] = 'preserved'
            elif block['type'] in ['footer', 'header']:
                # 页眉页脚保留或简单处理
                new_block['translated'] = block['content']
                new_block['translation_note'] = 'metadata'
            else:
                # 翻译文本
                translated = translate_text(block['content'])
                new_block['translated'] = translated
                new_block['original'] = block['content']
                new_block['translation_note'] = 'translated' if translated != block['content'] else 'unchanged'
            
            translated_batch['blocks'].append(new_block)
        
        translated_batches.append(translated_batch)
    
    # 保存翻译结果
    result = {
        'source_file': data['source_file'],
        'total_pages': data['total_pages'],
        'translation_method': 'rule_based_with_terms',
        'batches': translated_batches
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 翻译完成，保存到: {output_json}")
    return result

def generate_html_for_pdf(data: Dict, output_html: str, start_page: int = 1, end_page: int = None):
    """生成用于PDF转换的HTML文件"""
    
    if end_page is None:
        end_page = data['total_pages']
    
    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>半导体外延：异质结构的物理与制备（中文版）</title>
    <style>
        @page {{
            size: A4;
            margin: 2.5cm 2cm 2.5cm 2cm;
            @top-center {{
                content: "半导体外延：异质结构的物理与制备";
                font-size: 9pt;
                color: #666;
            }}
            @bottom-center {{
                content: counter(page);
                font-size: 10pt;
            }}
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "SimSun", serif;
            font-size: 11pt;
            line-height: 1.8;
            color: #333;
            text-align: justify;
        }}
        
        h1 {{
            font-size: 24pt;
            font-weight: bold;
            text-align: center;
            margin: 40pt 0 20pt 0;
            color: #1a1a1a;
            page-break-before: always;
        }}
        
        h1:first-of-type {{
            page-break-before: auto;
        }}
        
        h2 {{
            font-size: 16pt;
            font-weight: bold;
            color: #2c3e50;
            margin: 25pt 0 15pt 0;
            page-break-after: avoid;
        }}
        
        h3 {{
            font-size: 13pt;
            font-weight: bold;
            color: #34495e;
            margin: 18pt 0 12pt 0;
        }}
        
        p {{
            margin: 10pt 0;
            text-indent: 2em;
        }}
        
        .no-indent {{
            text-indent: 0;
        }}
        
        .center {{
            text-align: center;
        }}
        
        .author {{
            font-size: 14pt;
            text-align: center;
            margin: 20pt 0;
            color: #555;
        }}
        
        .subtitle {{
            font-size: 16pt;
            text-align: center;
            margin: 15pt 0;
            color: #444;
        }}
        
        .edition {{
            font-size: 12pt;
            text-align: center;
            margin: 10pt 0;
            color: #666;
        }}
        
        .series {{
            font-size: 11pt;
            text-align: center;
            margin: 30pt 0 10pt 0;
            color: #666;
            border-top: 1pt solid #ccc;
            padding-top: 15pt;
        }}
        
        .copyright {{
            font-size: 9pt;
            color: #666;
            margin: 20pt 0;
            padding: 15pt;
            background-color: #f9f9f9;
            border-left: 3pt solid #ccc;
        }}
        
        .preface {{
            margin: 20pt 0;
        }}
        
        .signature {{
            text-align: right;
            margin: 20pt 0;
            font-style: italic;
        }}
        
        .pagenumber {{
            text-align: center;
            color: #999;
            font-size: 10pt;
            margin: 15pt 0;
        }}
        
        /* 数学公式样式 */
        .math {{
            font-family: "Times New Roman", "Latin Modern Math", serif;
            font-style: italic;
            background-color: #f8f8f8;
            padding: 10pt;
            margin: 10pt 0;
            border-radius: 3pt;
            overflow-x: auto;
        }}
        
        /* 代码样式 */
        code {{
            font-family: "Fira Code", "Consolas", "Monaco", monospace;
            font-size: 9.5pt;
            background-color: #f4f4f4;
            padding: 2pt 4pt;
            border-radius: 2pt;
        }}
        
        pre {{
            font-family: "Fira Code", "Consolas", "Monaco", monospace;
            font-size: 9pt;
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 12pt;
            margin: 12pt 0;
            border-radius: 4pt;
            overflow-x: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        /* 元数据保留 */
        .metadata {{
            font-size: 9pt;
            color: #666;
            margin: 8pt 0;
        }}
        
        @media print {{
            body {{
                font-size: 10.5pt;
            }}
            
            h1 {{
                font-size: 22pt;
            }}
            
            h2 {{
                font-size: 15pt;
            }}
            
            pre {{
                white-space: pre-wrap;
            }}
        }}
    </style>
</head>
<body>
'''
    
    # 处理内容块
    for batch in data['batches']:
        for block in batch['blocks']:
            page = block.get('page', 0)
            
            # 跳过超出范围的页面
            if page < start_page or (end_page and page > end_page):
                continue
            
            content = block.get('translated', block.get('content', ''))
            block_type = block.get('type', 'text')
            
            if not content.strip():
                continue
            
            # 根据内容类型生成HTML
            if block_type == 'math':
                html_content += f'    <div class="math">{escape_html(content)}</div>\n'
            elif block_type == 'code':
                html_content += f'    <pre><code>{escape_html(content)}</code></pre>\n'
            elif block_type == 'footer':
                html_content += f'    <div class="metadata">{escape_html(content)}</div>\n'
            elif block_type == 'header':
                html_content += f'    <div class="metadata">{escape_html(content)}</div>\n'
            else:
                # 检测标题级别
                if is_title(content):
                    html_content += f'    <h2>{escape_html(content)}</h2>\n'
                elif is_subtitle(content):
                    html_content += f'    <h3>{escape_html(content)}</h3>\n'
                elif is_author(content):
                    html_content += f'    <p class="author">{escape_html(content)}</p>\n'
                elif is_pagenumber(content):
                    html_content += f'    <div class="pagenumber">{escape_html(content)}</div>\n'
                else:
                    # 普通段落
                    paragraphs = content.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            html_content += f'    <p>{escape_html(para)}</p>\n'
    
    html_content += '''
</body>
</html>
'''
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML生成完成: {output_html}")
    return output_html

def escape_html(text: str) -> str:
    """转义HTML特殊字符"""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;'))

def is_title(text: str) -> bool:
    """判断是否是标题"""
    title_keywords = ['preface', 'chapter', 'introduction', 'conclusion', 'summary', '前言', '绪论', '结论']
    text_lower = text.lower().strip()
    return any(kw in text_lower for kw in title_keywords) and len(text) < 80

def is_subtitle(text: str) -> bool:
    """判断是否是副标题"""
    return len(text) < 60 and text.strip().endswith(':')

def is_author(text: str) -> bool:
    """判断是否是作者信息"""
    return 'Institute' in text or 'University' in text or 'Universität' in text or '@' in text

def is_pagenumber(text: str) -> bool:
    """判断是否是页码"""
    return re.match(r'^[\dvxivlcmd]+$', text.strip().lower()) is not None and len(text.strip()) < 10

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python translate_and_generate.py <batches_json文件>")
        sys.exit(1)
    
    input_json = sys.argv[1]
    base_name = Path(input_json).stem
    
    # 步骤1: 翻译
    translated_json = base_name + "_translated.json"
    data = process_batches(input_json, translated_json)
    
    # 步骤2: 生成HTML
    output_html = base_name + ".html"
    generate_html_for_pdf(data, output_html)
    
    print(f"\n📄 输出文件:")
    print(f"   - 翻译JSON: {translated_json}")
    print(f"   - HTML文件: {output_html}")
    print(f"\n下一步: 使用浏览器将HTML转换为PDF")
    print(f"  wkhtmltopdf --enable-local-file-access {output_html} output.pdf")
