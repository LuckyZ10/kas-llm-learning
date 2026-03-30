#!/usr/bin/env python3
"""
PDF英译中工具 - 保留代码与数学公式
适用于学术书籍翻译
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
from dataclasses import dataclass

@dataclass
class TextBlock:
    """文本块，区分普通文本、代码、数学公式"""
    content: str
    block_type: str  # 'text', 'code', 'math', 'header', 'footer'
    page_num: int
    
class PDFTranslator:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        
    def detect_block_type(self, text: str, font_info: Dict) -> str:
        """检测文本块类型"""
        # 检测数学公式 - 包含希腊字母、上下标、积分符号等
        math_patterns = [
            r'[α-ωΑ-Ω]',  # 希腊字母
            r'[∫∑∏√∞±×÷]',  # 数学符号
            r'_\{[^}]+\}',  # 下标
            r'\^\{[^}]+\}',  # 上标
            r'\$[^$]+\$',  # LaTeX格式
            r'\\[a-zA-Z]+\{',  # LaTeX命令
            r'[=<>≤≥≈≠±]\s*[-+]?\d+\.?\d*',  # 等式和数字
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, text):
                return 'math'
        
        # 检测代码 - 多行缩进、编程关键词等
        code_patterns = [
            r'^(def|class|import|from|if|for|while|return)\s',
            r'[{;]\s*$',  # 以分号或大括号结尾
            r'^(\s{4,}|\t)',  # 缩进
            r'(function|var|let|const|public|private)\s+\w+',
        ]
        
        code_score = sum(1 for p in code_patterns if re.search(p, text, re.MULTILINE))
        if code_score >= 2 or ('    ' in text and len(text.split('\n')) > 3):
            return 'code'
        
        # 检测页眉页脚
        if len(text) < 100 and any(x in text.lower() for x in ['page', 'chapter', '©', 'springer']):
            if font_info.get('size', 12) < 10:
                return 'header' if 'page' in text.lower() else 'footer'
        
        return 'text'
    
    def extract_page(self, page_num: int) -> List[TextBlock]:
        """提取单页内容并分类"""
        page = self.doc[page_num]
        blocks = []
        
        # 获取文本块
        text_dict = page.get_text("dict")
        
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue
                
            block_text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"]
                block_text += "\n"
            
            if not block_text.strip():
                continue
                
            # 获取字体信息
            font_info = {
                'name': block["lines"][0]["spans"][0].get("font", ""),
                'size': block["lines"][0]["spans"][0].get("size", 12),
                'flags': block["lines"][0]["spans"][0].get("flags", 0)
            }
            
            block_type = self.detect_block_type(block_text, font_info)
            blocks.append(TextBlock(
                content=block_text.strip(),
                block_type=block_type,
                page_num=page_num + 1
            ))
        
        return blocks
    
    def prepare_translation_batches(self, start_page: int = 0, end_page: int = None, batch_size: int = 10) -> List[Dict]:
        """准备翻译批次"""
        if end_page is None:
            end_page = self.total_pages
            
        batches = []
        for i in range(start_page, end_page, batch_size):
            batch_end = min(i + batch_size, end_page)
            batch = {
                'start_page': i + 1,
                'end_page': batch_end,
                'blocks': []
            }
            
            for page_num in range(i, batch_end):
                blocks = self.extract_page(page_num)
                batch['blocks'].extend([
                    {
                        'content': b.content,
                        'type': b.block_type,
                        'page': b.page_num
                    }
                    for b in blocks
                ])
            
            batches.append(batch)
            
        return batches
    
    def save_batches_json(self, output_path: str, start_page: int = 0, end_page: int = None, batch_size: int = 10):
        """保存批次到JSON供翻译"""
        batches = self.prepare_translation_batches(start_page, end_page, batch_size)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'source_file': str(self.pdf_path),
                'total_pages': self.total_pages,
                'batches': batches
            }, f, ensure_ascii=False, indent=2)
            
        print(f"已提取 {len(batches)} 个批次，保存到 {output_path}")
        return batches


def translate_batch(batch: Dict, target_lang: str = "zh-CN") -> Dict:
    """
    翻译单个批次
    保留 code 和 math 类型不翻译
    """
    translated = batch.copy()
    translated['blocks'] = []
    
    for block in batch['blocks']:
        new_block = block.copy()
        
        if block['type'] in ['code', 'math']:
            # 保留原样
            new_block['translated'] = block['content']
            new_block['translated_type'] = 'preserved'
        elif block['type'] in ['header', 'footer']:
            # 页眉页脚简单处理或保留
            new_block['translated'] = block['content']
            new_block['translated_type'] = 'metadata'
        else:
            # 标记为待翻译
            new_block['translated'] = None
            new_block['translated_type'] = 'pending'
            
        translated['blocks'].append(new_block)
        
    return translated


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python pdf_translator.py <pdf文件路径> [起始页] [结束页]")
        print("示例: python pdf_translator.py book.pdf 1 50")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    start_page = int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 0
    end_page = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    translator = PDFTranslator(pdf_path)
    
    # 提取并保存
    output_json = Path(pdf_path).stem + "_batches.json"
    batches = translator.save_batches_json(output_json, start_page, end_page, batch_size=5)
    
    print(f"\nPDF共 {translator.total_pages} 页")
    print(f"已提取批次: {len(batches)}")
    print(f"输出文件: {output_json}")
