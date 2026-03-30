"""
Core exceptions for HPC Connector.
"""


class HPCConnectorError(Exception):
    """Base exception for all HPC connector errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "HPC_ERROR"
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"[{self.error_code}] {self.message} - Details: {self.details}"
        return f"[{self.error_code}] {self.message}"


class AuthenticationError(HPCConnectorError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "AUTH_ERROR", details)


class ConnectionError(HPCConnectorError):
    """Raised when connection to cluster fails."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONN_ERROR", details)


class JobSubmissionError(HPCConnectorError):
    """Raised when job submission fails."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "JOB_SUBMIT_ERROR", details)


class JobMonitorError(HPCConnectorError):
    """Raised when job monitoring fails."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "JOB_MONITOR_ERROR", details)


class DataTransferError(HPCConnectorError):
    """Raised when data transfer fails."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "DATA_TRANSFER_ERROR", details)


class ResourceError(HPCConnectorError):
    """Raised when resource allocation fails."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "RESOURCE_ERROR", details)


class ConfigurationError(HPCConnectorError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "CONFIG_ERROR", details)


class RecoveryError(HPCConnectorError):
    """Raised when fault recovery fails."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "RECOVERY_ERROR", details)


class TimeoutError(HPCConnectorError):
    """Raised when operation times out."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, "TIMEOUT_ERROR", details)
