"""
数据模型定义
Pydantic models for API Gateway
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, EmailStr, validator


# ==================== 用户认证模型 ====================

class UserRole(str, Enum):
    """用户角色"""
    ADMIN = "admin"
    RESEARCHER = "researcher"
    GUEST = "guest"
    API_CLIENT = "api_client"


class UserStatus(str, Enum):
    """用户状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class UserBase(BaseModel):
    """用户基础模型"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole = UserRole.RESEARCHER
    department: Optional[str] = None
    institution: Optional[str] = None


class UserCreate(UserBase):
    """创建用户请求"""
    password: str = Field(..., min_length=8, max_length=100)


class UserUpdate(BaseModel):
    """更新用户请求"""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    department: Optional[str] = None
    institution: Optional[str] = None
    status: Optional[UserStatus] = None


class UserInDB(UserBase):
    """数据库中的用户模型"""
    id: int
    hashed_password: str
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    api_key: Optional[str] = None
    quota_limit: int = 1000  # 每日请求配额
    quota_used: int = 0

    class Config:
        from_attributes = True


class UserResponse(UserBase):
    """用户响应模型"""
    id: int
    status: UserStatus
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    """JWT令牌响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600  # 默认1小时


class TokenPayload(BaseModel):
    """JWT令牌payload"""
    sub: str  # 用户ID
    exp: datetime
    iat: datetime
    role: str
    scope: List[str] = []


class LoginRequest(BaseModel):
    """登录请求"""
    username: str
    password: str
    grant_type: Optional[str] = "password"


class PasswordChange(BaseModel):
    """密码修改请求"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


# ==================== API Key模型 ====================

class APIKeyCreate(BaseModel):
    """创建API Key请求"""
    name: str = Field(..., max_length=100)
    description: Optional[str] = None
    expires_in_days: Optional[int] = 365
    scopes: List[str] = ["read"]


class APIKeyResponse(BaseModel):
    """API Key响应"""
    id: int
    name: str
    key: str  # 仅创建时返回
    prefix: str
    description: Optional[str]
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    is_active: bool


# ==================== 任务模型 ====================

class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(int, Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class TaskType(str, Enum):
    """任务类型"""
    DFT_CALCULATION = "dft_calculation"
    MD_SIMULATION = "md_simulation"
    ML_TRAINING = "ml_training"
    ACTIVE_LEARNING = "active_learning"
    SCREENING = "screening"
    ANALYSIS = "analysis"
    CONVERSION = "conversion"
    EXPORT = "export"


class TaskBase(BaseModel):
    """任务基础模型"""
    name: str = Field(..., max_length=200)
    task_type: TaskType
    priority: TaskPriority = TaskPriority.NORMAL
    description: Optional[str] = None


class TaskCreate(TaskBase):
    """创建任务请求"""
    input_data: Dict[str, Any]
    timeout: Optional[int] = 3600  # 默认1小时超时
    callback_url: Optional[str] = None


class TaskUpdate(BaseModel):
    """更新任务请求"""
    name: Optional[str] = None
    priority: Optional[TaskPriority] = None
    status: Optional[TaskStatus] = None


class TaskResponse(BaseModel):
    """任务响应模型"""
    id: str
    name: str
    task_type: TaskType
    status: TaskStatus
    priority: TaskPriority
    progress: float = Field(0.0, ge=0.0, le=100.0)
    created_by: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None  # 秒

    class Config:
        from_attributes = True


class TaskListResponse(BaseModel):
    """任务列表响应"""
    total: int
    items: List[TaskResponse]
    page: int
    page_size: int


class TaskQueueStatus(BaseModel):
    """任务队列状态"""
    queue_name: str
    pending_count: int
    running_count: int
    completed_count: int
    failed_count: int
    avg_wait_time: float  # 平均等待时间(秒)
    avg_execution_time: float  # 平均执行时间(秒)


# ==================== DFT计算模型 ====================

class DFTCalculationType(str, Enum):
    """DFT计算类型"""
    SCF = "scf"
    RELAX = "relax"
    BANDS = "bands"
    DOS = "dos"
    PHONON = "phonon"
    NEB = "neb"
    MD = "md"
    OPTIMIZE = "optimize"


class DFTCode(str, Enum):
    """DFT代码"""
    VASP = "vasp"
    QUANTUM_ESPRESSO = "quantum_espresso"
    ABACUS = "abacus"
    CP2K = "cp2k"


class DFTCalculationRequest(BaseModel):
    """DFT计算请求"""
    name: str
    calculation_type: DFTCalculationType
    code: DFTCode = DFTCode.VASP
    structure: Dict[str, Any]  # POSCAR/结构数据
    parameters: Dict[str, Any]  # 计算参数
    kpoints: Optional[Dict[str, Any]] = None
    pseudopotentials: Optional[Dict[str, str]] = None
    priority: TaskPriority = TaskPriority.NORMAL


class DFTCalculationResponse(BaseModel):
    """DFT计算响应"""
    task_id: str
    status: TaskStatus
    message: str


# ==================== MD模拟模型 ====================

class MDSimulationType(str, Enum):
    """MD模拟类型"""
    NVT = "nvt"
    NPT = "npt"
    NVE = "nve"
    LANGEVIN = "langevin"
    METADYNAMICS = "metadynamics"


class MDPotential(str, Enum):
    """MD势能"""
    NEP = "nep"
    DEEPMD = "deepmd"
    REAXFF = "reaxff"
    EAM = "eam"
    MEAM = "meam"
    SNAP = "snap"


class MDSimulationRequest(BaseModel):
    """MD模拟请求"""
    name: str
    simulation_type: MDSimulationType
    potential: MDPotential
    structure: Dict[str, Any]
    temperature: float = Field(..., gt=0)
    pressure: Optional[float] = None
    time_step: float = 1.0  # fs
    n_steps: int = Field(..., gt=0)
    ensemble: Optional[str] = None
    constraints: Optional[List[Dict[str, Any]]] = None
    priority: TaskPriority = TaskPriority.NORMAL


class MDSimulationResponse(BaseModel):
    """MD模拟响应"""
    task_id: str
    status: TaskStatus
    message: str


# ==================== ML训练模型 ====================

class MLModelType(str, Enum):
    """ML模型类型"""
    NEP = "nep"
    DEEPMD = "deepmd"
    GAP = "gap"
    ACE = "ace"
    SCHNET = "schnet"
    MEGNET = "megnet"


class MLTrainingRequest(BaseModel):
    """ML训练请求"""
    name: str
    model_type: MLModelType
    training_data: Dict[str, Any]  # 训练数据集
    validation_data: Optional[Dict[str, Any]] = None
    hyperparameters: Dict[str, Any]
    epochs: int = 1000
    batch_size: int = 32
    device: str = "auto"  # auto/cpu/cuda
    priority: TaskPriority = TaskPriority.NORMAL


class MLTrainingResponse(BaseModel):
    """ML训练响应"""
    task_id: str
    status: TaskStatus
    message: str


# ==================== 筛选模型 ====================

class ScreeningCriteria(BaseModel):
    """筛选条件"""
    property_name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    weight: float = 1.0


class ScreeningRequest(BaseModel):
    """筛选请求"""
    name: str
    dataset: Dict[str, Any]  # 数据集
    criteria: List[ScreeningCriteria]
    method: str = "multi_objective"  # multi_objective/pareto/clustering
    top_k: int = 10
    priority: TaskPriority = TaskPriority.NORMAL


class ScreeningResponse(BaseModel):
    """筛选响应"""
    task_id: str
    status: TaskStatus
    message: str


# ==================== 系统监控模型 ====================

class SystemMetrics(BaseModel):
    """系统指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_tasks: int
    queued_tasks: int
    api_requests_per_min: int
    avg_response_time_ms: float


class APIMetrics(BaseModel):
    """API指标"""
    endpoint: str
    method: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    errors_by_status: Dict[int, int]


class HealthStatus(BaseModel):
    """健康状态"""
    status: str  # healthy/degraded/unhealthy
    version: str
    uptime_seconds: float
    checks: Dict[str, bool]
    timestamp: datetime


# ==================== 文件模型 ====================

class FileUploadResponse(BaseModel):
    """文件上传响应"""
    file_id: str
    filename: str
    size: int
    content_type: str
    uploaded_at: datetime
    url: str


class FileInfo(BaseModel):
    """文件信息"""
    file_id: str
    filename: str
    size: int
    content_type: str
    uploaded_at: datetime
    uploaded_by: int
    metadata: Optional[Dict[str, Any]] = None


# ==================== 响应包装模型 ====================

class APIResponse(BaseModel):
    """标准API响应"""
    success: bool
    message: str
    data: Optional[Any] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class PaginatedResponse(BaseModel):
    """分页响应"""
    success: bool = True
    message: str = "success"
    data: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """错误响应"""
    success: bool = False
    message: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# ==================== 配置模型 ====================

class GatewayConfig(BaseModel):
    """网关配置"""
    debug: bool = False
    secret_key: str
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    algorithm: str = "HS256"
    rate_limit_per_minute: int = 100
    max_upload_size_mb: int = 100
    celery_broker_url: str
    celery_result_backend: str
    database_url: str
    redis_url: str
    log_level: str = "INFO"
    allowed_hosts: List[str] = ["*"]
    cors_origins: List[str] = ["*"]
