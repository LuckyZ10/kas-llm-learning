"""
KAS Security Module
Sensitive information detection and filtering
Resource quota management and monitoring
Network access control
Docker sandbox isolation
Secure sandbox integration
"""
from kas.core.security.sensitive_filter import (
    SensitiveInfoFilter,
    FilterConfig,
    MatchInfo,
    get_default_filter,
    filter_text,
    is_sensitive,
    detect_sensitive,
)
from kas.core.security.resource_quota import (
    ResourceQuota,
    QuotaStatus,
    ResourceMonitor,
    ResourceLimiter,
    ResourceStatus,
    with_quota,
    create_default_quota,
    create_strict_quota,
    create_relaxed_quota,
)
from kas.core.security.network_controller import (
    NetworkPolicy,
    NetworkAccessController,
    NetworkInterceptor,
    NetworkMode,
    PolicyPreset,
    create_strict_controller,
    create_moderate_controller,
    create_permissive_controller,
    get_default_controller,
    set_default_controller,
)
from kas.core.security.docker_sandbox import (
    DockerSandbox,
    DockerSandboxConfig,
    DockerSandboxManager,
    ExecutionResult,
    DockerNotAvailableError,
    SandboxCreationError,
    SandboxExecutionError,
    SandboxTimeoutError,
)
from kas.core.security.secure_sandbox import (
    SecureSandbox,
    SecureSandboxConfig,
    SecureSandboxManager,
    SecureExecutionResult,
    create_secure_sandbox,
)

__all__ = [
    # Sensitive filter
    "SensitiveInfoFilter",
    "FilterConfig",
    "MatchInfo",
    "get_default_filter",
    "filter_text",
    "is_sensitive",
    "detect_sensitive",
    # Resource quota
    "ResourceQuota",
    "QuotaStatus",
    "ResourceMonitor",
    "ResourceLimiter",
    "ResourceStatus",
    "with_quota",
    "create_default_quota",
    "create_strict_quota",
    "create_relaxed_quota",
    # Network controller
    "NetworkPolicy",
    "NetworkAccessController",
    "NetworkInterceptor",
    "NetworkMode",
    "PolicyPreset",
    "create_strict_controller",
    "create_moderate_controller",
    "create_permissive_controller",
    "get_default_controller",
    "set_default_controller",
    # Docker sandbox
    "DockerSandbox",
    "DockerSandboxConfig",
    "DockerSandboxManager",
    "ExecutionResult",
    "DockerNotAvailableError",
    "SandboxCreationError",
    "SandboxExecutionError",
    "SandboxTimeoutError",
    # Secure sandbox (integrated)
    "SecureSandbox",
    "SecureSandboxConfig",
    "SecureSandboxManager",
    "SecureExecutionResult",
    "create_secure_sandbox",
]
