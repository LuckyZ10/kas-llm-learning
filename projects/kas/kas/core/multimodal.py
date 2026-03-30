"""
KAS 多模态输入处理
支持文件上传、预处理，自动路由到对应装备
"""
import os
import io
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, BinaryIO, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FileType(Enum):
    """支持的文件类型"""
    PDF = "pdf"
    IMAGE = "image"  # jpg, png, gif, webp, etc.
    TEXT = "text"    # txt, md, json, yaml, csv, etc.
    CODE = "code"    # py, js, ts, html, css, etc.
    DOCUMENT = "document"  # docx, odt, etc.
    SPREADSHEET = "spreadsheet"  # xlsx, csv, etc.
    UNKNOWN = "unknown"


@dataclass
class ProcessedFile:
    """处理后的文件信息"""
    original_name: str
    file_type: FileType
    mime_type: str
    size: int
    hash: str  # SHA256
    content: Any  # 根据类型不同，可能是 str, bytes, dict
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Attachment:
    """附件信息"""
    name: str
    content: Union[bytes, str]
    content_type: str
    size: int


class FileProcessor(ABC):
    """文件处理器基类"""
    
    supported_types: List[FileType] = []
    supported_extensions: List[str] = []
    
    @abstractmethod
    def can_process(self, file_path: str, mime_type: str = "") -> bool:
        """检查是否能处理该文件"""
        pass
    
    @abstractmethod
    def process(self, file_data: Union[bytes, BinaryIO], filename: str) -> ProcessedFile:
        """处理文件，返回结构化内容"""
        pass
    
    def _calculate_hash(self, data: bytes) -> str:
        """计算文件哈希"""
        return hashlib.sha256(data).hexdigest()[:16]


class PDFProcessor(FileProcessor):
    """PDF 处理器 - 路由到 pdf_parser 装备"""
    
    supported_types = [FileType.PDF]
    supported_extensions = ['.pdf']
    
    def can_process(self, file_path: str, mime_type: str = "") -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions or mime_type == "application/pdf"
    
    def process(self, file_data: Union[bytes, BinaryIO], filename: str) -> ProcessedFile:
        """处理 PDF 文件"""
        # 读取数据
        if hasattr(file_data, 'read'):
            data = file_data.read()
        else:
            data = file_data
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        file_hash = self._calculate_hash(data)
        
        # 尝试使用 pdf_parser 装备
        try:
            from kas.core.equipment import get_equipment_pool
            
            pool = get_equipment_pool()
            result = pool.use("pdf_parser", {
                "file_data": data,
                "filename": filename
            })
            
            return ProcessedFile(
                original_name=filename,
                file_type=FileType.PDF,
                mime_type="application/pdf",
                size=len(data),
                hash=file_hash,
                content=result.get("text", ""),
                metadata={
                    "pages": result.get("pages", 0),
                    "title": result.get("metadata", {}).get("title", ""),
                    "author": result.get("metadata", {}).get("author", ""),
                },
                processing_info={
                    "method": "pdf_parser_equipment",
                    "success": True
                }
            )
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            
            # Fallback: 返回原始内容提示
            return ProcessedFile(
                original_name=filename,
                file_type=FileType.PDF,
                mime_type="application/pdf",
                size=len(data),
                hash=file_hash,
                content=f"[PDF file: {filename}, size: {len(data)} bytes]",
                metadata={},
                processing_info={
                    "method": "fallback",
                    "success": False,
                    "error": str(e)
                }
            )


class ImageProcessor(FileProcessor):
    """图片处理器 - 路由到 ocr 或 image_analysis 装备"""
    
    supported_types = [FileType.IMAGE]
    supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff']
    
    MIME_TYPES = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
    }
    
    def can_process(self, file_path: str, mime_type: str = "") -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions or mime_type.startswith("image/")
    
    def process(self, file_data: Union[bytes, BinaryIO], filename: str, use_ocr: bool = True) -> ProcessedFile:
        """处理图片文件"""
        # 读取数据
        if hasattr(file_data, 'read'):
            data = file_data.read()
        else:
            data = file_data
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        file_hash = self._calculate_hash(data)
        ext = Path(filename).suffix.lower()
        mime_type = self.MIME_TYPES.get(ext, 'image/unknown')
        
        result = {
            "text": "",
            "description": "",
            "ocr_success": False
        }
        
        # 尝试 OCR
        if use_ocr:
            try:
                from kas.core.equipment import get_equipment_pool
                
                pool = get_equipment_pool()
                ocr_result = pool.use("ocr", {
                    "image_data": data,
                })
                
                result["text"] = ocr_result.get("text", "")
                result["ocr_success"] = True
                result["ocr_confidence"] = ocr_result.get("confidence", 0)
                
            except Exception as e:
                logger.warning(f"OCR failed: {e}")
                result["ocr_error"] = str(e)
        
        return ProcessedFile(
            original_name=filename,
            file_type=FileType.IMAGE,
            mime_type=mime_type,
            size=len(data),
            hash=file_hash,
            content=result["text"] if result["ocr_success"] else f"[Image: {filename}]",
            metadata={
                "width": None,  # 可以添加图片尺寸
                "height": None,
                "ocr_text": result["text"] if result["ocr_success"] else None,
                "ocr_confidence": result.get("ocr_confidence", 0),
            },
            processing_info={
                "method": "ocr_equipment" if result["ocr_success"] else "raw",
                "ocr_success": result["ocr_success"],
                "error": result.get("ocr_error")
            }
        )


class TextProcessor(FileProcessor):
    """文本处理器 - 直接读取或路由到 file_reader"""
    
    supported_types = [FileType.TEXT, FileType.CODE]
    supported_extensions = [
        '.txt', '.md', '.markdown',
        '.json', '.yaml', '.yml',
        '.csv', '.tsv',
        '.py', '.js', '.ts', '.jsx', '.tsx',
        '.html', '.htm', '.css', '.scss', '.sass',
        '.java', '.c', '.cpp', '.h', '.hpp',
        '.go', '.rs', '.rb', '.php',
        '.sh', '.bash', '.zsh',
        '.sql', '.xml', '.toml'
    ]
    
    ENCODINGS = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
    
    def can_process(self, file_path: str, mime_type: str = "") -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions or mime_type.startswith("text/")
    
    def process(self, file_data: Union[bytes, BinaryIO, str], filename: str) -> ProcessedFile:
        """处理文本/代码文件"""
        # 读取数据
        if hasattr(file_data, 'read'):
            data = file_data.read()
        else:
            data = file_data
        
        # 如果是字符串，直接处理
        if isinstance(data, str):
            text_content = data
            raw_bytes = data.encode('utf-8')
        else:
            raw_bytes = data
            # 尝试多种编码
            text_content = self._decode_text(data)
        
        file_hash = self._calculate_hash(raw_bytes)
        ext = Path(filename).suffix.lower()
        
        # 判断文件类型
        if ext in ['.py', '.js', '.ts', '.java', '.go', '.rs', '.c', '.cpp']:
            file_type = FileType.CODE
            mime_type = f"text/x-{ext[1:]}"
        else:
            file_type = FileType.TEXT
            mime_type = "text/plain"
        
        # 尝试解析结构化数据
        parsed_content = None
        if ext == '.json':
            try:
                import json
                parsed_content = json.loads(text_content)
            except Exception:
                pass
        elif ext in ['.yaml', '.yml']:
            try:
                import yaml
                parsed_content = yaml.safe_load(text_content)
            except Exception:
                pass
        
        return ProcessedFile(
            original_name=filename,
            file_type=file_type,
            mime_type=mime_type,
            size=len(raw_bytes),
            hash=file_hash,
            content=text_content,
            metadata={
                "lines": text_content.count('\n') + 1,
                "parsed": parsed_content is not None,
                "parsed_content": parsed_content
            },
            processing_info={
                "method": "text_decode",
                "encoding": "detected"  # 实际检测到的编码
            }
        )
    
    def _decode_text(self, data: bytes) -> str:
        """尝试多种编码解码文本"""
        for encoding in self.ENCODINGS:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # 最后尝试忽略错误
        return data.decode('utf-8', errors='ignore')


class MultimodalProcessor:
    """多模态处理器 - 统一入口"""
    
    def __init__(self):
        self.processors: List[FileProcessor] = [
            PDFProcessor(),
            ImageProcessor(),
            TextProcessor(),
        ]
    
    def process_file(self, file_path: str) -> ProcessedFile:
        """处理文件路径"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 读取文件
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # 找到合适的处理器
        processor = self._find_processor(file_path, "")
        
        if processor:
            return processor.process(data, path.name)
        else:
            # 未知类型，返回原始信息
            return ProcessedFile(
                original_name=path.name,
                file_type=FileType.UNKNOWN,
                mime_type="application/octet-stream",
                size=len(data),
                hash=hashlib.sha256(data).hexdigest()[:16],
                content=f"[Binary file: {path.name}, size: {len(data)} bytes]",
                metadata={},
                processing_info={"method": "raw_binary", "success": True}
            )
    
    def process_upload(self, filename: str, content: bytes, content_type: str = "") -> ProcessedFile:
        """处理上传的文件内容"""
        processor = self._find_processor(filename, content_type)
        
        if processor:
            return processor.process(content, filename)
        else:
            return ProcessedFile(
                original_name=filename,
                file_type=FileType.UNKNOWN,
                mime_type=content_type or "application/octet-stream",
                size=len(content),
                hash=hashlib.sha256(content).hexdigest()[:16],
                content=f"[Binary file: {filename}, size: {len(content)} bytes]",
                metadata={},
                processing_info={"method": "raw_binary", "success": True}
            )
    
    def _find_processor(self, file_path: str, mime_type: str) -> Optional[FileProcessor]:
        """找到能处理该文件的处理器"""
        for processor in self.processors:
            if processor.can_process(file_path, mime_type):
                return processor
        return None
    
    def process_attachments(self, attachments: List[Attachment]) -> List[ProcessedFile]:
        """批量处理附件"""
        results = []
        for attachment in attachments:
            try:
                processed = self.process_upload(
                    attachment.name,
                    attachment.content if isinstance(attachment.content, bytes) else attachment.content.encode(),
                    attachment.content_type
                )
                results.append(processed)
            except Exception as e:
                logger.error(f"Failed to process attachment {attachment.name}: {e}")
                # 添加失败的占位符
                results.append(ProcessedFile(
                    original_name=attachment.name,
                    file_type=FileType.UNKNOWN,
                    mime_type="",
                    size=attachment.size,
                    hash="",
                    content=f"[Failed to process: {str(e)}]",
                    metadata={},
                    processing_info={"error": str(e), "success": False}
                ))
        return results


# ==================== 与 Agent/Chat 集成 ====================

class MultimodalChat:
    """支持多模态输入的对话"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.processor = MultimodalProcessor()
    
    def run(self, message: str, attachments: List[Attachment] = None, use_mock: bool = False) -> str:
        """运行对话，支持附件"""
        # 处理附件
        processed_files = []
        if attachments:
            processed_files = self.processor.process_attachments(attachments)
        
        # 构建增强提示词
        enhanced_message = self._build_multimodal_prompt(message, processed_files)
        
        # 调用普通对话
        from kas.core.chat import ChatEngine
        chat = ChatEngine(self.agent_name)
        return chat.run(enhanced_message, use_mock=use_mock)
    
    def _build_multimodal_prompt(self, message: str, files: List[ProcessedFile]) -> str:
        """构建包含文件内容的多模态提示词"""
        if not files:
            return message
        
        prompt_parts = [message, "\n\n## 附件内容\n"]
        
        for i, file in enumerate(files, 1):
            prompt_parts.append(f"\n### 附件 {i}: {file.original_name}\n")
            prompt_parts.append(f"类型: {file.file_type.value}, 大小: {file.size} bytes\n")
            
            if file.file_type == FileType.IMAGE:
                # 图片：添加 OCR 结果
                ocr_text = file.metadata.get("ocr_text")
                if ocr_text:
                    prompt_parts.append(f"图片文字识别结果:\n{ocr_text}\n")
                else:
                    prompt_parts.append(f"[图片文件，无法识别文字]\n")
            
            elif file.file_type == FileType.PDF:
                # PDF：添加提取的文本
                prompt_parts.append(f"PDF 内容:\n{file.content[:2000]}...\n" if len(file.content) > 2000 else f"PDF 内容:\n{file.content}\n")
            
            elif file.file_type in [FileType.TEXT, FileType.CODE]:
                # 文本/代码：添加内容
                content_preview = file.content[:3000] if len(file.content) > 3000 else file.content
                prompt_parts.append(f"文件内容:\n```\n{content_preview}\n```\n")
                if len(file.content) > 3000:
                    prompt_parts.append(f"[文件已截断，共 {len(file.content)} 字符]\n")
            
            else:
                # 其他：添加基本信息
                prompt_parts.append(f"[二进制文件，无法显示内容]\n")
        
        return "".join(prompt_parts)


# ==================== CLI 工具函数 ====================

def process_file_for_agent(file_path: str, agent_name: str = None) -> ProcessedFile:
    """处理文件供 Agent 使用"""
    processor = MultimodalProcessor()
    return processor.process_file(file_path)


def format_file_for_prompt(processed: ProcessedFile) -> str:
    """将处理后的文件格式化为提示词文本"""
    lines = [f"\n## 文件: {processed.original_name}\n"]
    lines.append(f"类型: {processed.file_type.value}\n")
    lines.append(f"大小: {processed.size} bytes\n")
    
    if processed.file_type == FileType.IMAGE:
        ocr_text = processed.metadata.get("ocr_text")
        if ocr_text:
            lines.append(f"\n图片中的文字:\n{ocr_text}\n")
        else:
            lines.append("\n[图片文件]\n")
    
    elif processed.file_type == FileType.PDF:
        lines.append(f"\nPDF 内容:\n{processed.content[:2000]}...\n" if len(processed.content) > 2000 else f"\nPDF 内容:\n{processed.content}\n")
    
    elif processed.file_type in [FileType.TEXT, FileType.CODE]:
        content = processed.content[:3000]
        lines.append(f"\n```\n{content}\n```\n")
        if len(processed.content) > 3000:
            lines.append(f"[文件已截断，共 {len(processed.content)} 字符]\n")
    
    return "".join(lines)
