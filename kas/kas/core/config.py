"""
KAS Core - Configuration Management
配置管理系统 - 像手机设置一样管理 KAS

功能：
- 本地配置文件管理 (~/.kas/config.yaml)
- API Key 本地存储（安全，不上传 git）
- 多 LLM 提供商配置
- 用户偏好设置
- 配置验证和默认值
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


# KAS 配置目录
KAS_DIR = Path.home() / '.kas'
CONFIG_FILE = KAS_DIR / 'config.yaml'
CREDENTIALS_FILE = KAS_DIR / 'credentials.yaml'


@dataclass
class LLMConfig:
    """LLM 配置"""
    provider: str = "deepseek"  # deepseek/kimi/openai
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2000
    base_url: Optional[str] = None
    # API Key 不存这里，存 credentials.yaml


@dataclass
class IngestionConfig:
    """代码吞食配置"""
    default_output_dir: str = "~/.kas/agents"
    max_file_size: int = 1024 * 1024  # 1MB
    exclude_patterns: list = field(default_factory=lambda: [
        "node_modules", "__pycache__", ".git", "dist", "build"
    ])
    include_extensions: list = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".java", ".go", ".rs", ".md", ".rst"
    ])


@dataclass
class ChatConfig:
    """对话配置"""
    default_agent: str = ""
    save_history: bool = True
    history_limit: int = 100
    auto_evaluate: bool = True  # 自动评估每次对话质量


@dataclass
class EvolutionConfig:
    """进化配置"""
    auto_evolve: bool = False  # 是否自动触发进化
    convergence_threshold: float = 0.85
    quality_threshold: float = 70.0
    max_generations: int = 10


@dataclass
class CloudConfig:
    """云端市场配置"""
    api_url: str = "http://localhost:8000"
    enabled: bool = True
    auto_sync: bool = False  # 是否自动同步到云端


@dataclass
class KASConfig:
    """KAS 全局配置"""
    version: str = "0.1.0"
    llm: LLMConfig = field(default_factory=LLMConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """
    配置管理器 - 像手机设置一样简单
    
    用法：
        config = ConfigManager()
        
        # 读取配置
        model = config.llm.model
        
        # 修改配置
        config.llm.temperature = 0.5
        config.save()
        
        # 获取 API Key（自动从 credentials 读取）
        api_key = config.get_api_key()
    """
    
    def __init__(self):
        self.config = KASConfig()
        self.credentials: Dict[str, str] = {}
        self._ensure_dirs()
        self._load()
    
    def _ensure_dirs(self):
        """确保配置目录存在"""
        KAS_DIR.mkdir(parents=True, exist_ok=True)
        (KAS_DIR / 'agents').mkdir(exist_ok=True)
        (KAS_DIR / 'versions').mkdir(exist_ok=True)
        (KAS_DIR / 'learning').mkdir(exist_ok=True)
    
    def _load(self):
        """加载配置"""
        # 加载主配置
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                self.config = self._dict_to_config(data)
        
        # 加载凭证（API Keys）
        if CREDENTIALS_FILE.exists():
            with open(CREDENTIALS_FILE, 'r', encoding='utf-8') as f:
                self.credentials = yaml.safe_load(f) or {}
    
    def save(self):
        """保存配置（不含 API Keys）"""
        # 保存主配置
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self.config), f, allow_unicode=True, default_flow_style=False)
        
        # 保存凭证（单独文件，已被 .gitignore 保护）
        self._save_credentials()
    
    def _save_credentials(self):
        """保存凭证（安全存储）"""
        with open(CREDENTIALS_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(self.credentials, f, allow_unicode=True, default_flow_style=False)
        
        # 设置文件权限（仅限当前用户读取）
        os.chmod(CREDENTIALS_FILE, 0o600)
    
    def _dict_to_config(self, data: Dict) -> KASConfig:
        """字典转配置对象"""
        return KASConfig(
            version=data.get('version', '0.1.0'),
            llm=LLMConfig(**data.get('llm', {})),
            ingestion=IngestionConfig(**data.get('ingestion', {})),
            chat=ChatConfig(**data.get('chat', {})),
            evolution=EvolutionConfig(**data.get('evolution', {})),
            cloud=CloudConfig(**data.get('cloud', {})),
            user_preferences=data.get('user_preferences', {})
        )
    
    # ========== 快捷访问 ==========
    
    @property
    def llm(self) -> LLMConfig:
        return self.config.llm
    
    @property
    def ingestion(self) -> IngestionConfig:
        return self.config.ingestion
    
    @property
    def chat(self) -> ChatConfig:
        return self.config.chat
    
    @property
    def evolution(self) -> EvolutionConfig:
        return self.config.evolution
    
    @property
    def cloud(self) -> CloudConfig:
        return self.config.cloud
    
    @property
    def agents_dir(self) -> Path:
        """Agent存储目录"""
        return KAS_DIR / 'agents'
    
    @property
    def config_dir(self) -> Path:
        """配置目录"""
        return KAS_DIR
    
    # ========== API Key 管理 ==========
    
    def set_api_key(self, provider: str, api_key: str):
        """
        设置 API Key（安全存储，不上传 git）
        
        支持的 provider:
        - deepseek
        - kimi
        - openai
        """
        self.credentials[f'{provider}_api_key'] = api_key
        self._save_credentials()
    
    def get_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """
        获取 API Key
        
        优先顺序：
        1. 指定的 provider
        2. 配置中的默认 provider
        3. 环境变量
        """
        if provider is None:
            provider = self.config.llm.provider
        
        # 1. 从凭证文件读取
        key = self.credentials.get(f'{provider}_api_key')
        if key:
            return key
        
        # 2. 从环境变量读取
        env_map = {
            'deepseek': 'DEEPSEEK_API_KEY',
            'kimi': 'KIMI_API_KEY',
            'openai': 'OPENAI_API_KEY'
        }
        env_var = env_map.get(provider)
        if env_var:
            return os.environ.get(env_var)
        
        return None
    
    def list_configured_providers(self) -> list:
        """列出已配置 API Key 的提供商"""
        providers = []
        for provider in ['deepseek', 'kimi', 'openai']:
            if self.get_api_key(provider):
                providers.append(provider)
        return providers
    
    # ========== 用户偏好 ==========
    
    def set_preference(self, key: str, value: Any):
        """设置用户偏好"""
        self.config.user_preferences[key] = value
        self.save()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """获取用户偏好"""
        return self.config.user_preferences.get(key, default)
    
    # ========== 配置验证 ==========
    
    def validate(self) -> list:
        """验证配置是否完整"""
        errors = []
        
        # 检查是否有 LLM 配置
        if not self.list_configured_providers():
            errors.append("未配置任何 LLM API Key，运行 'kas config setup' 设置")
        
        return errors
    
    def get_status(self) -> str:
        """获取配置状态（用于显示）"""
        providers = self.list_configured_providers()
        
        lines = [
            "📋 KAS 配置状态",
            "=" * 40,
            f"版本: {self.config.version}",
            f"LLM 提供商: {self.config.llm.provider}",
            f"已配置 API: {', '.join(providers) if providers else '❌ 未配置'}",
            f"默认模型: {self.config.llm.model}",
            f"温度参数: {self.config.llm.temperature}",
            f"最大 token: {self.config.llm.max_tokens}",
            "",
            "📁 配置文件位置:",
            f"  主配置: {CONFIG_FILE}",
            f"  凭证: {CREDENTIALS_FILE} (安全)",
        ]
        
        return "\n".join(lines)


# 全局配置实例（单例模式）
_config_instance = None

def get_config() -> ConfigManager:
    """获取全局配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance


def init_config():
    """初始化配置（首次使用）"""
    config = get_config()
    
    # 检查是否需要初始化
    if not CONFIG_FILE.exists():
        print("🚀 首次使用 KAS，正在初始化配置...")
        config.save()
        print(f"✅ 配置已保存到: {CONFIG_FILE}")
        print(f"🔐 API Key 将安全存储在: {CREDENTIALS_FILE}")
        print("   (该文件已被 .gitignore 保护，不会上传)")
    
    return config


# 辅助函数
def setup_wizard():
    """
    交互式配置向导
    
    引导用户完成初始配置
    """
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    
    console = Console()
    config = get_config()
    
    console.print("\n🚀 KAS 配置向导\n", style="bold blue")
    
    # 选择 LLM 提供商
    provider = Prompt.ask(
        "选择默认 LLM 提供商",
        choices=["deepseek", "kimi", "openai"],
        default="deepseek"
    )
    config.llm.provider = provider
    
    # 设置 API Key
    api_key = Prompt.ask(
        f"输入你的 {provider} API Key",
        password=True
    )
    if api_key:
        config.set_api_key(provider, api_key)
        console.print(f"✅ {provider} API Key 已安全保存", style="green")
    
    console.print("\n☁️ 云端市场配置", style="bold blue")
    
    # 云端市场设置
    config.cloud.enabled = Confirm.ask(
        "是否启用云端市场？",
        default=True
    )
    
    if config.cloud.enabled:
        cloud_url = Prompt.ask(
            "云端市场 API 地址",
            default="http://localhost:8000"
        )
        config.cloud.api_url = cloud_url
        
        # 云端 API Key（可选）
        cloud_key = Prompt.ask(
            "云端市场 API Key (可选，发布/评分需要)",
            password=True
        )
        if cloud_key:
            config.credentials['cloud_api_key'] = cloud_key
            console.print("✅ 云端 API Key 已保存", style="green")
    
    # 其他设置
    config.llm.temperature = float(Prompt.ask(
        "设置回答创造性 (0.0-1.0)",
        default="0.7"
    ))
    
    config.chat.save_history = Confirm.ask(
        "是否保存对话历史？",
        default=True
    )
    
    # 保存
    config.save()
    config._save_credentials()  # 确保凭证也保存
    
    console.print(f"\n✅ 配置完成！", style="bold green")
    console.print(f"📁 配置文件: {CONFIG_FILE}")
    console.print(f"🔐 凭证文件: {CREDENTIALS_FILE}")
    console.print("\n你可以随时运行 'kas config' 修改设置")
    
    return config


if __name__ == "__main__":
    # 测试
    config = init_config()
    print(config.get_status())
    
    # 测试 API Key（如果有）
    key = config.get_api_key()
    if key:
        print(f"\n✅ 找到 API Key: {key[:20]}...")
    else:
        print("\n⚠️ 未找到 API Key，请运行 setup_wizard()")
