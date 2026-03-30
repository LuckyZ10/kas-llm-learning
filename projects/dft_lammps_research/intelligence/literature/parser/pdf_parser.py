"""
PDF文献解析器
"""

import re
import os
import io
import tempfile
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

from ..config.models import Paper
from ..config.settings import DATA_SOURCES, PAPERS_DIR


@dataclass
class ParsedSection:
    """解析的章节"""
    title: str
    content: str
    level: int = 1


@dataclass
class ParsedTable:
    """解析的表格"""
    caption: str
    headers: List[str]
    rows: List[List[str]]
    page: int = 0


@dataclass
class ParsedFigure:
    """解析的图表"""
    caption: str
    figure_type: str  # image, diagram, chart
    page: int = 0
    description: str = ""


class PDFParser:
    """PDF解析器"""
    
    def __init__(self):
        self.papers_dir = PAPERS_DIR
        self.papers_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_paper(self, paper: Paper, pdf_content: Optional[bytes] = None) -> Paper:
        """
        解析论文PDF
        
        Args:
            paper: 论文对象
            pdf_content: PDF内容，如果为None则尝试从pdf_url下载
        
        Returns:
            更新后的论文对象
        """
        # 获取PDF内容
        if pdf_content is None:
            pdf_content = self._download_pdf(paper)
        
        if not pdf_content:
            print(f"无法获取PDF: {paper.title}")
            return paper
        
        try:
            # 解析PDF
            text = self._extract_text(pdf_content)
            sections = self._extract_sections(text)
            
            # 更新论文对象
            paper.full_text = text
            paper.sections = {s.title: s.content for s in sections}
            
            # 提取方法信息
            methods = self._extract_methods(text)
            software = self._extract_software(text)
            
            paper.methods = methods
            paper.software = software
            
            # 保存PDF到本地
            self._save_pdf(paper, pdf_content)
            
        except Exception as e:
            print(f"解析PDF失败: {e}")
        
        return paper
    
    def _download_pdf(self, paper: Paper) -> Optional[bytes]:
        """下载PDF"""
        import urllib.request
        
        urls_to_try = []
        
        if paper.pdf_url:
            urls_to_try.append(paper.pdf_url)
        
        # arXiv特殊处理
        if paper.arxiv_id:
            urls_to_try.append(f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf")
        
        for url in urls_to_try:
            try:
                headers = {
                    "User-Agent": "LiteratureSurveyBot/1.0 (research@example.com)"
                }
                req = urllib.request.Request(url, headers=headers)
                
                with urllib.request.urlopen(req, timeout=30) as response:
                    content = response.read()
                    if len(content) > 1000:  # 确保是有效的PDF
                        return content
            except Exception as e:
                print(f"下载PDF失败 ({url}): {e}")
                continue
        
        return None
    
    def _extract_text(self, pdf_content: bytes) -> str:
        """提取PDF文本"""
        try:
            # 尝试使用PyPDF2
            import PyPDF2
            
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text_parts = []
            
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception:
                    continue
            
            return "\n\n".join(text_parts)
        
        except ImportError:
            # 备用方案：使用pdfplumber
            try:
                import pdfplumber
                
                text_parts = []
                with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                
                return "\n\n".join(text_parts)
            
            except ImportError:
                raise ImportError("需要安装PyPDF2或pdfplumber: pip install PyPDF2 pdfplumber")
    
    def _extract_sections(self, text: str) -> List[ParsedSection]:
        """提取章节结构"""
        sections = []
        
        # 常见章节标题模式
        section_patterns = [
            r'\n\s*(\d+)\.\s+([A-Z][^\n]+)\n',  # 1. Introduction
            r'\n\s*([A-Z][A-Z\s]+)\s*\n',  # INTRODUCTION
            r'\n\s*(Abstract|Introduction|Methods|Results|Discussion|Conclusion|References)\s*\n',
            r'\n\s*(Background|Methodology|Experimental|Computational Details|Theory)\s*\n'
        ]
        
        # 尝试匹配章节
        section_starts = []
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                section_starts.append((match.start(), match.group(0).strip(), 1))
        
        # 排序并按位置去重
        section_starts.sort(key=lambda x: x[0])
        
        # 构建章节
        for i, (start, title, level) in enumerate(section_starts):
            end = section_starts[i + 1][0] if i + 1 < len(section_starts) else len(text)
            content = text[start:end].strip()
            
            # 清理标题
            title = title.strip()
            title = re.sub(r'^\d+\.\s*', '', title)  # 移除编号
            
            sections.append(ParsedSection(title=title, content=content, level=level))
        
        # 如果没有找到章节，创建默认结构
        if not sections and text:
            # 尝试提取摘要
            abstract_match = re.search(
                r'Abstract[.:]?\s*(.+?)(?=\n\s*(?:Introduction|I\.\s|1\.\s|\Z))',
                text,
                re.DOTALL | re.IGNORECASE
            )
            
            if abstract_match:
                sections.append(ParsedSection(
                    title="Abstract",
                    content=abstract_match.group(1).strip()
                ))
            
            # 剩余内容作为正文
            remaining = text
            if abstract_match:
                remaining = text[abstract_match.end():]
            
            if remaining.strip():
                sections.append(ParsedSection(
                    title="Full Text",
                    content=remaining.strip()
                ))
        
        return sections
    
    def _extract_methods(self, text: str) -> List[str]:
        """提取方法信息"""
        methods = []
        config = DATA_SOURCES.get("analysis", {}).get("method_extraction", {})
        method_patterns = config.get("method_patterns", [])
        
        text_lower = text.lower()
        
        for pattern in method_patterns:
            if pattern.lower() in text_lower:
                # 找到完整词
                for match in re.finditer(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE):
                    method = match.group(0)
                    if method not in methods:
                        methods.append(method)
        
        # 额外提取计算细节
        computational_patterns = [
            r'\b(VASP|Quantum ESPRESSO|Gaussian|ABINIT|CASTEP)\b',
            r'\b(DFT|DFT-D[23]?|vdW-DF|SCAN|HSE06|PBE)\b',
            r'\b(LAMMPS|GROMACS|NAMD)\b',
            r'\b(PAW|USPP|norm-conserving)\b',
            r'\b(k-point|energy cutoff|convergence)\w*\s+criteria?\b'
        ]
        
        for pattern in computational_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                method = match.group(0)
                if method not in methods:
                    methods.append(method)
        
        return methods
    
    def _extract_software(self, text: str) -> List[str]:
        """提取软件信息"""
        software = []
        config = DATA_SOURCES.get("analysis", {}).get("method_extraction", {})
        software_patterns = config.get("software_patterns", [])
        
        text_lower = text.lower()
        
        for pattern in software_patterns:
            if pattern.lower() in text_lower:
                # 检查是否是完整词
                if re.search(r'\b' + re.escape(pattern) + r'\b', text, re.IGNORECASE):
                    if pattern not in software:
                        software.append(pattern)
        
        return software
    
    def _save_pdf(self, paper: Paper, pdf_content: bytes):
        """保存PDF到本地"""
        # 生成文件名
        safe_title = re.sub(r'[^\w\s-]', '', paper.title)[:50]
        filename = f"{paper.id.replace(':', '_')}_{safe_title}.pdf"
        filepath = self.papers_dir / filename
        
        try:
            with open(filepath, 'wb') as f:
                f.write(pdf_content)
            print(f"PDF已保存: {filepath}")
        except Exception as e:
            print(f"保存PDF失败: {e}")
    
    def extract_tables(self, pdf_content: bytes) -> List[ParsedTable]:
        """提取表格"""
        tables = []
        
        try:
            import pdfplumber
            
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()
                    
                    for table_data in page_tables:
                        if table_data and len(table_data) > 1:
                            table = ParsedTable(
                                caption="",
                                headers=table_data[0] if table_data else [],
                                rows=table_data[1:] if len(table_data) > 1 else [],
                                page=page_num
                            )
                            tables.append(table)
        
        except ImportError:
            print("需要安装pdfplumber: pip install pdfplumber")
        
        return tables
    
    def extract_figures(self, pdf_content: bytes) -> List[ParsedFigure]:
        """提取图表信息（基于文本分析）"""
        figures = []
        
        # 提取图表标题
        text = self._extract_text(pdf_content)
        
        # 匹配图表标题
        figure_patterns = [
            r'Figure\s+(\d+)[:.]?\s*(.+?)(?=\n|Figure|\Z)',
            r'Fig\.?\s+(\d+)[:.]?\s*(.+?)(?=\n|Fig\.?|Figure|\Z)',
            r'Table\s+(\d+)[:.]?\s*(.+?)(?=\n|Table|\Z)'
        ]
        
        for pattern in figure_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                fig_num = match.group(1)
                caption = match.group(2).strip()
                
                fig_type = "chart" if "table" in pattern.lower() else "figure"
                
                figure = ParsedFigure(
                    caption=f"Figure {fig_num}: {caption}",
                    figure_type=fig_type,
                    description=caption
                )
                figures.append(figure)
        
        return figures


class TextExtractor:
    """纯文本提取器（无需PDF库）"""
    
    @staticmethod
    def extract_from_html(html_content: str) -> str:
        """从HTML提取文本"""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除脚本和样式
            for script in soup(["script", "style"]):
                script.decompose()
            
            return soup.get_text(separator='\n', strip=True)
        
        except ImportError:
            # 简单HTML标签移除
            import re
            text = re.sub(r'<[^\u003e]+>', ' ', html_content)
            return " ".join(text.split())
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本"""
        # 移除多余空白
        text = " ".join(text.split())
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        return text.strip()
