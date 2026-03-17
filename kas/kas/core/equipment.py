"""
KAS 装备系统
提供 Agent 可使用的工具装备，支持 MCP 和 Plugin 两种类型
"""
import os
import json
import subprocess
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import select
import time

logger = logging.getLogger(__name__)


@dataclass
class EquipmentConfig:
    """装备配置"""
    name: str
    type: str  # 'mcp' or 'plugin'
    config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    version: str = "1.0.0"


class Equipment(ABC):
    """装备基类"""
    
    def __init__(self, config: EquipmentConfig):
        self.config = config
        self.name = config.name
        self.type = config.type
        self._initialized = False
    
    @abstractmethod
    def use(self, params: Dict[str, Any]) -> Any:
        """使用装备执行操作"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化装备，返回是否成功"""
        pass
    
    def is_available(self) -> bool:
        """检查装备是否可用"""
        return self._initialized
    
    def get_schema(self) -> Dict[str, Any]:
        """获取装备参数 schema"""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.config.description,
            "config": self.config.config
        }


class MCPEquipment(Equipment):
    """MCP (Model Context Protocol) 装备
    
    通过 stdio 或 HTTP 与 MCP Server 通信
    """
    
    def __init__(self, config: EquipmentConfig):
        super().__init__(config)
        self.server_command = config.config.get("command")
        self.server_args = config.config.get("args", [])
        self.env_vars = config.config.get("env", {})
        self._process = None
    
    def initialize(self) -> bool:
        """初始化 MCP Server，带超时控制"""
        try:
            if not self.server_command:
                logger.error(f"MCP equipment {self.name}: no command specified")
                return False
            
            # 检查命令是否存在
            cmd_path = self._find_command(self.server_command)
            if not cmd_path:
                logger.error(f"MCP equipment {self.name}: command not found: {self.server_command}")
                return False
            
            # 设置环境变量
            env = os.environ.copy()
            env.update(self.env_vars)
            
            # 启动 MCP Server
            self._process = subprocess.Popen(
                [cmd_path] + self.server_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # 发送初始化请求，带超时控制
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "kas", "version": "0.2.0"}
                }
            }
            
            self._send_request(init_request)
            
            # 读取响应，带 10 秒超时
            response = self._read_response(timeout=10.0)
            
            if response and "result" in response:
                self._initialized = True
                logger.info(f"MCP equipment {self.name} initialized successfully")
                return True
            else:
                logger.error(f"MCP equipment {self.name} initialization failed: {response}")
                self.shutdown()  # 清理失败的进程
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"MCP equipment {self.name}: initialization timeout")
            self.shutdown()
            return False
        except Exception as e:
            logger.error(f"MCP equipment {self.name} initialization error: {e}")
            self.shutdown()
            return False
    
    def use(self, params: Dict[str, Any]) -> Any:
        """调用 MCP 工具，带超时控制"""
        if not self._initialized:
            raise RuntimeError(f"Equipment {self.name} not initialized")
        
        tool_name = params.get("tool")
        tool_params = params.get("params", {})
        timeout = params.get("timeout", 30.0)  # 默认 30 秒超时
        
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_params
            }
        }
        
        self._send_request(request)
        response = self._read_response(timeout=timeout)
        
        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        else:
            raise RuntimeError(f"No response from MCP server (timeout: {timeout}s)")
    
    def _find_command(self, cmd: str) -> Optional[str]:
        """查找命令路径 - 使用 shutil.which() 更可靠"""
        # 先检查是否是绝对路径
        if os.path.isabs(cmd) and os.path.isfile(cmd):
            return cmd
        
        # 使用 shutil.which() 查找命令 (支持 Windows/Unix)
        cmd_path = shutil.which(cmd)
        if cmd_path:
            return cmd_path
        
        return None
    
    def _send_request(self, request: dict):
        """发送请求到 MCP Server"""
        if self._process and self._process.stdin:
            message = json.dumps(request) + "\n"
            self._process.stdin.write(message)
            self._process.stdin.flush()
    
    def _read_response(self, timeout: float = 30.0) -> Optional[dict]:
        """从 MCP Server 读取响应，带超时控制"""
        if not self._process or not self._process.stdout:
            return None
        
        try:
            # 使用 select 实现超时读取
            import select
            ready, _, _ = select.select([self._process.stdout], [], [], timeout)
            
            if ready:
                line = self._process.stdout.readline()
                if line:
                    return json.loads(line.strip())
            else:
                # 超时
                logger.warning(f"MCP equipment {self.name}: read response timeout ({timeout}s)")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"MCP equipment {self.name}: invalid JSON response - {e}")
            return None
        except Exception as e:
            logger.error(f"MCP equipment {self.name}: read error - {e}")
            return None
    
    def _next_id(self) -> int:
        """生成下一个请求 ID"""
        import itertools
        if not hasattr(self, '_id_counter'):
            self._id_counter = itertools.count(1)
        return next(self._id_counter)
    
    def shutdown(self):
        """关闭 MCP Server"""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self._initialized = False


class PluginEquipment(Equipment):
    """插件装备 - 内置 Python 实现"""
    
    def __init__(self, config: EquipmentConfig):
        super().__init__(config)
        self._handler: Optional[Callable] = None
    
    def initialize(self) -> bool:
        """初始化插件"""
        try:
            # 插件在初始化时不需要特别处理
            # 实际功能在 use 方法中实现
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Plugin equipment {self.name} initialization error: {e}")
            return False
    
    def use(self, params: Dict[str, Any]) -> Any:
        """调用插件功能"""
        if not self._initialized:
            raise RuntimeError(f"Equipment {self.name} not initialized")
        
        if self._handler:
            return self._handler(params)
        else:
            raise NotImplementedError(f"Plugin {self.name} handler not set")
    
    def set_handler(self, handler: Callable):
        """设置插件处理函数"""
        self._handler = handler


# ==================== 内置装备实现 ====================

class WebSearchEquipment(PluginEquipment):
    """网页搜索装备"""
    
    def __init__(self, config: EquipmentConfig):
        super().__init__(config)
        self.engine = config.config.get("engine", "duckduckgo")
    
    def initialize(self) -> bool:
        """初始化搜索装备"""
        try:
            # 检查依赖
            if self.engine == "duckduckgo":
                try:
                    import duckduckgo_search
                except ImportError:
                    logger.warning("duckduckgo-search not installed, using requests fallback")
            
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"WebSearch initialization error: {e}")
            return False
    
    def use(self, params: Dict[str, Any]) -> Any:
        """执行搜索"""
        query = params.get("query", "")
        max_results = params.get("max_results", 5)
        
        if self.engine == "duckduckgo":
            return self._search_duckduckgo(query, max_results)
        else:
            raise ValueError(f"Unsupported search engine: {self.engine}")
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict]:
        """使用 DuckDuckGo 搜索"""
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
                return results
        except ImportError:
            # Fallback: 使用 requests 调用 DuckDuckGo HTML
            return self._search_duckduckgo_fallback(query, max_results)
    
    def _search_duckduckgo_fallback(self, query: str, max_results: int) -> List[Dict]:
        """DuckDuckGo 搜索备用方案"""
        import requests
        from urllib.parse import quote
        
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        # 简单解析 HTML (实际使用应该上 BeautifulSoup)
        results = []
        # ... 解析逻辑 ...
        
        return results


class OCREquipment(PluginEquipment):
    """OCR 文字识别装备"""
    
    def __init__(self, config: EquipmentConfig):
        super().__init__(config)
        self.language = config.config.get("language", "en")
        self._processor = None
    
    def initialize(self) -> bool:
        """初始化 OCR"""
        try:
            # 尝试使用 pytesseract
            try:
                import pytesseract
                from PIL import Image
                self._processor = "tesseract"
            except ImportError:
                logger.warning("pytesseract not installed, OCR will use mock mode")
                self._processor = "mock"
            
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"OCR initialization error: {e}")
            return False
    
    def use(self, params: Dict[str, Any]) -> Any:
        """执行 OCR"""
        image_path = params.get("image")
        image_data = params.get("image_data")
        
        if not image_path and not image_data:
            raise ValueError("Either image path or image_data must be provided")
        
        if self._processor == "tesseract":
            return self._ocr_tesseract(image_path or image_data)
        else:
            # Mock mode
            return {
                "text": f"[OCR Mock] Recognized text from {image_path or 'image_data'}",
                "confidence": 0.95,
                "language": self.language
            }
    
    def _ocr_tesseract(self, image_input) -> Dict:
        """使用 Tesseract OCR"""
        import pytesseract
        from PIL import Image
        
        if isinstance(image_input, str) and os.path.isfile(image_input):
            image = Image.open(image_input)
        else:
            # 处理 image_data (bytes)
            from io import BytesIO
            image = Image.open(BytesIO(image_input))
        
        text = pytesseract.image_to_string(image, lang=self.language)
        
        return {
            "text": text,
            "confidence": 0.9,  # Tesseract 不提供简单置信度
            "language": self.language
        }


class PDFParserEquipment(PluginEquipment):
    """PDF 解析装备"""
    
    def __init__(self, config: EquipmentConfig):
        super().__init__(config)
        self.extract_images = config.config.get("extract_images", False)
    
    def initialize(self) -> bool:
        """初始化 PDF 解析器"""
        try:
            try:
                import PyPDF2
                self._parser = "pypdf2"
            except ImportError:
                try:
                    import pdfplumber
                    self._parser = "pdfplumber"
                except ImportError:
                    logger.warning("No PDF library installed, using mock mode")
                    self._parser = "mock"
            
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"PDF parser initialization error: {e}")
            return False
    
    def use(self, params: Dict[str, Any]) -> Any:
        """解析 PDF"""
        file_path = params.get("file")
        file_data = params.get("file_data")
        pages = params.get("pages")  # 指定页码，None表示全部
        
        if not file_path and not file_data:
            raise ValueError("Either file path or file_data must be provided")
        
        if self._parser == "pypdf2":
            return self._parse_pypdf2(file_path or file_data, pages)
        elif self._parser == "pdfplumber":
            return self._parse_pdfplumber(file_path or file_data, pages)
        else:
            # Mock mode
            return {
                "text": f"[PDF Mock] Parsed content from {file_path or 'file_data'}",
                "pages": 5,
                "metadata": {"title": "Mock PDF", "author": "Mock"}
            }
    
    def _parse_pypdf2(self, file_input, pages: Optional[List[int]]) -> Dict:
        """使用 PyPDF2 解析"""
        import PyPDF2
        
        if isinstance(file_input, str):
            with open(file_input, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return self._extract_pdf_content(reader, pages)
        else:
            from io import BytesIO
            reader = PyPDF2.PdfReader(BytesIO(file_input))
            return self._extract_pdf_content(reader, pages)
    
    def _parse_pdfplumber(self, file_input, pages: Optional[List[int]]) -> Dict:
        """使用 pdfplumber 解析"""
        import pdfplumber
        
        if isinstance(file_input, str):
            pdf = pdfplumber.open(file_input)
        else:
            from io import BytesIO
            pdf = pdfplumber.open(BytesIO(file_input))
        
        text_parts = []
        page_count = len(pdf.pages)
        
        page_range = pages or range(page_count)
        for i in page_range:
            if 0 <= i < page_count:
                page = pdf.pages[i]
                text_parts.append(page.extract_text() or "")
        
        pdf.close()
        
        return {
            "text": "\n\n".join(text_parts),
            "pages": page_count,
            "metadata": {}
        }
    
    def _extract_pdf_content(self, reader, pages: Optional[List[int]]) -> Dict:
        """提取 PDF 内容"""
        text_parts = []
        page_count = len(reader.pages)
        
        page_range = pages or range(page_count)
        for i in page_range:
            if 0 <= i < page_count:
                page = reader.pages[i]
                text_parts.append(page.extract_text() or "")
        
        return {
            "text": "\n\n".join(text_parts),
            "pages": page_count,
            "metadata": reader.metadata or {}
        }


class FileReaderEquipment(PluginEquipment):
    """文件读取装备 - 支持多种格式"""
    
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.md': 'markdown',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.csv': 'csv',
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.html': 'html',
        '.css': 'css',
    }
    
    def initialize(self) -> bool:
        self._initialized = True
        return True
    
    def use(self, params: Dict[str, Any]) -> Any:
        """读取文件"""
        file_path = params.get("file")
        content = params.get("content")  # 直接传入内容
        encoding = params.get("encoding", "utf-8")
        
        if content:
            return self._parse_content(content, params.get("filename", "unknown"))
        
        if not file_path:
            raise ValueError("Either file path or content must be provided")
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 读取文件
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()
        
        return self._parse_content(content, file_path)
    
    def _parse_content(self, content: str, filename: str) -> Dict:
        """根据文件类型解析内容"""
        ext = Path(filename).suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(ext, 'text')
        
        result = {
            "content": content,
            "type": file_type,
            "filename": filename,
            "size": len(content)
        }
        
        # 特殊处理
        if file_type == 'json':
            try:
                result["parsed"] = json.loads(content)
            except json.JSONDecodeError:
                pass
        elif file_type == 'yaml':
            try:
                import yaml
                result["parsed"] = yaml.safe_load(content)
            except ImportError:
                pass
        
        return result


class CodeExecutorEquipment(PluginEquipment):
    """代码执行装备 (沙箱模式)"""
    
    def __init__(self, config: EquipmentConfig):
        super().__init__(config)
        self.sandbox = config.config.get("sandbox", "none")  # none, docker, subprocess
        self.timeout = config.config.get("timeout", 30)
        self.allowed_languages = config.config.get("languages", ["python"])
    
    def initialize(self) -> bool:
        """初始化代码执行器"""
        if self.sandbox == "docker":
            # 检查 Docker 是否可用
            try:
                subprocess.run(["docker", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Docker not available, falling back to subprocess sandbox")
                self.sandbox = "subprocess"
        
        self._initialized = True
        return True
    
    def use(self, params: Dict[str, Any]) -> Any:
        """执行代码"""
        code = params.get("code")
        language = params.get("language", "python")
        inputs = params.get("inputs", {})
        
        if not code:
            raise ValueError("Code must be provided")
        
        if language not in self.allowed_languages:
            raise ValueError(f"Language {language} not allowed")
        
        if self.sandbox == "docker":
            return self._execute_docker(code, language, inputs)
        elif self.sandbox == "subprocess":
            return self._execute_subprocess(code, language, inputs)
        else:
            return self._execute_unsafe(code, language, inputs)
    
    def _execute_subprocess(self, code: str, language: str, inputs: dict) -> Dict:
        """使用子进程执行 (有限沙箱)"""
        if language != "python":
            raise ValueError(f"Subprocess sandbox only supports Python")
        
        # 创建临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # 执行代码
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={"PATH": "/usr/bin:/bin"}  # 限制环境
            )
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timeout after {self.timeout}s",
                "returncode": -1,
                "success": False
            }
        finally:
            os.unlink(temp_file)
    
    def _execute_docker(self, code: str, language: str, inputs: dict) -> Dict:
        """使用 Docker 执行 (完整沙箱)"""
        import tempfile
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # 使用 Docker 运行
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{temp_file}:/code/script.py:ro",
                    "-w", "/code",
                    "python:3.11-slim",
                    "python", "script.py"
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timeout after {self.timeout}s",
                "returncode": -1,
                "success": False
            }
        finally:
            os.unlink(temp_file)
    
    def _execute_unsafe(self, code: str, language: str, inputs: dict) -> Dict:
        """直接执行 (仅用于开发/测试)"""
        logger.warning("Executing code without sandbox - SECURITY RISK!")
        
        stdout_buffer = []
        stderr_buffer = []
        
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        
        try:
            if language == "python":
                exec(code, {"__builtins__": __builtins__}, inputs)
            
            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue()
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": 0,
                "success": True
            }
        except Exception as e:
            return {
                "stdout": sys.stdout.getvalue(),
                "stderr": f"{type(e).__name__}: {str(e)}",
                "returncode": 1,
                "success": False
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# ==================== 装备池 ====================

class EquipmentPool:
    """装备池 - 管理 Crew 的所有共享装备"""
    
    def __init__(self):
        self._equipment: Dict[str, Equipment] = {}
        self._builtins_registered = False
    
    def register_builtin_equipment(self):
        """注册内置装备"""
        if self._builtins_registered:
            return
        
        # Web Search
        self.register(WebSearchEquipment(EquipmentConfig(
            name="web_search",
            type="plugin",
            config={"engine": "duckduckgo"},
            description="Search the web for information"
        )))
        
        # OCR
        self.register(OCREquipment(EquipmentConfig(
            name="ocr",
            type="plugin",
            config={"language": "zh+en"},
            description="Extract text from images using OCR"
        )))
        
        # PDF Parser
        self.register(PDFParserEquipment(EquipmentConfig(
            name="pdf_parser",
            type="plugin",
            config={"extract_images": False},
            description="Extract text from PDF files"
        )))
        
        # File Reader
        self.register(FileReaderEquipment(EquipmentConfig(
            name="file_reader",
            type="plugin",
            description="Read various file formats"
        )))
        
        # Code Executor
        self.register(CodeExecutorEquipment(EquipmentConfig(
            name="code_executor",
            type="plugin",
            config={"sandbox": "subprocess", "timeout": 30},
            description="Execute code in sandboxed environment"
        )))
        
        self._builtins_registered = True
        logger.info("Builtin equipment registered")
    
    def register(self, equipment: Equipment):
        """注册装备"""
        self._equipment[equipment.name] = equipment
        
        # 自动初始化
        if not equipment.is_available():
            equipment.initialize()
    
    def get(self, name: str) -> Optional[Equipment]:
        """获取装备"""
        return self._equipment.get(name)
    
    def list_all(self) -> List[Dict]:
        """列出所有装备"""
        return [eq.get_schema() for eq in self._equipment.values()]
    
    def use(self, name: str, params: Dict[str, Any]) -> Any:
        """使用装备"""
        equipment = self.get(name)
        if not equipment:
            raise ValueError(f"Equipment not found: {name}")
        
        if not equipment.is_available():
            raise RuntimeError(f"Equipment {name} not initialized")
        
        return equipment.use(params)
    
    def shutdown_all(self):
        """关闭所有装备"""
        for equipment in self._equipment.values():
            if isinstance(equipment, MCPEquipment):
                equipment.shutdown()


# 全局装备池实例
_global_pool: Optional[EquipmentPool] = None


def get_equipment_pool() -> EquipmentPool:
    """获取全局装备池"""
    global _global_pool
    if _global_pool is None:
        _global_pool = EquipmentPool()
        _global_pool.register_builtin_equipment()
    return _global_pool
