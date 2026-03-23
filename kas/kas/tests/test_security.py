"""
Tests for KAS Security Module
"""
import pytest
from pathlib import Path
import tempfile

from kas.core.security import (
    # Sensitive filter
    SensitiveInfoFilter,
    FilterConfig,
    MatchInfo,
    filter_text,
    is_sensitive,
    # Resource quota
    ResourceQuota,
    QuotaStatus,
    ResourceMonitor,
    ResourceLimiter,
    create_default_quota,
    # Network controller
    NetworkPolicy,
    NetworkAccessController,
    NetworkMode,
    PolicyPreset,
    # Secure sandbox
    SecureSandbox,
    SecureSandboxConfig,
    SecureExecutionResult,
    create_secure_sandbox,
)


class TestSensitiveInfoFilter:
    """Test sensitive information filter"""
    
    def test_filter_api_key(self):
        """Test API key filtering"""
        text = "My API key is sk-1234567890abcdef1234567890abcdef"
        result = filter_text(text)
        assert "sk-1234567890abcdef1234567890abcdef" not in result
        assert "****" in result
    
    def test_filter_password(self):
        """Test password filtering"""
        text = "password=secret123 and pwd: hidden456"
        result = filter_text(text)
        assert "secret123" not in result
        assert "hidden456" not in result
    
    def test_filter_database_connection(self):
        """Test database connection string filtering"""
        text = "mysql://admin:pass123@localhost:3306/db"
        result = filter_text(text)
        assert "pass123" not in result
    
    def test_detect_sensitive(self):
        """Test sensitive info detection"""
        text = "API key: sk-test123 and email: test@example.com"
        assert is_sensitive(text) == True
        
        text = "This is a normal text without secrets"
        assert is_sensitive(text) == False
    
    def test_custom_rule(self):
        """Test adding custom filter rule"""
        config = FilterConfig()
        filter_obj = SensitiveInfoFilter(config)
        filter_obj.add_rule(r"CUSTOM-\d+", "custom_id")
        
        text = "My ID is CUSTOM-12345"
        result = filter_obj.filter(text)
        assert "CUSTOM-12345" not in result
    
    def test_whitelist(self):
        """Test whitelist pattern"""
        config = FilterConfig(whitelist_patterns=[r"sk-test-.*"])
        filter_obj = SensitiveInfoFilter(config)
        
        text = "Use sk-test-demo for testing"
        result = filter_obj.filter(text)
        assert "sk-test-demo" in result


class TestResourceQuota:
    """Test resource quota management"""
    
    def test_default_quota_creation(self):
        """Test creating default quota"""
        quota = create_default_quota()
        assert quota.max_cpu_percent > 0
        assert quota.max_memory_mb > 0
        assert quota.max_execution_time > 0
    
    def test_quota_status(self):
        """Test quota status check"""
        quota = ResourceQuota(
            max_cpu_percent=100,
            max_memory_mb=1024,
            max_execution_time=60,
        )
        monitor = ResourceMonitor()
        status = monitor.check_quota(quota)
        
        assert status is not None
        assert isinstance(status, QuotaStatus)
    
    def test_resource_limiter(self):
        """Test resource limiter"""
        quota = ResourceQuota(
            max_cpu_percent=100,
            max_memory_mb=1024,
            max_execution_time=10,
        )
        limiter = ResourceLimiter(quota)
        
        def simple_func():
            return "done"
        
        result = limiter.execute(simple_func)
        assert result["success"] == True
        assert result["result"] == "done"


class TestNetworkAccessController:
    """Test network access controller"""
    
    def test_allow_all_mode(self):
        """Test allow all mode"""
        policy = NetworkPolicy(mode="allow_all")
        controller = NetworkAccessController(policy)
        
        assert controller.is_allowed("example.com", 443) == True
        assert controller.is_allowed("any.host", 80) == True
    
    def test_deny_all_mode(self):
        """Test deny all mode"""
        policy = NetworkPolicy(mode="deny_all")
        controller = NetworkAccessController(policy)
        
        assert controller.is_allowed("example.com", 443) == False
    
    def test_whitelist_mode(self):
        """Test whitelist mode"""
        policy = NetworkPolicy(
            mode="whitelist",
            allowed_hosts=["api.openai.com", "*.github.com"],
        )
        controller = NetworkAccessController(policy)
        
        assert controller.is_allowed("api.openai.com", 443) == True
        assert controller.is_allowed("github.com", 443) == True
        assert controller.is_allowed("api.github.com", 443) == True
        assert controller.is_allowed("other.com", 443) == False
    
    def test_strict_preset(self):
        """Test strict policy preset"""
        from kas.core.security import create_strict_controller
        controller = create_strict_controller()
        
        # Should allow common LLM API hosts
        assert controller.is_allowed("api.openai.com", 443) == True
        assert controller.is_allowed("api.anthropic.com", 443) == True


class TestSecureSandbox:
    """Test secure sandbox integration"""
    
    def test_config_creation(self):
        """Test secure sandbox config"""
        config = SecureSandboxConfig(
            name="test-sandbox",
            work_dir=Path("./test_workspace"),
        )
        
        assert config.name == "test-sandbox"
        assert config.enable_filter == True
        assert config.enable_quota == True
        assert config.enable_network_control == True
    
    def test_sandbox_creation(self):
        """Test creating secure sandbox"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = create_secure_sandbox(
                name="test",
                work_dir=tmpdir,
                preset="default",
                use_docker=False,  # Disable Docker for testing
            )
            
            assert sandbox is not None
            assert sandbox.name == "test"
    
    def test_sandbox_filter_output(self):
        """Test sandbox output filtering"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = create_secure_sandbox(
                name="test",
                work_dir=tmpdir,
                preset="default",
                use_docker=False,
            )
            
            output = "API key: sk-test123456789"
            filtered = sandbox.filter_text(output)
            assert "sk-test123456789" not in filtered
    
    def test_sandbox_context_manager(self):
        """Test sandbox as context manager"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with create_secure_sandbox(
                name="test",
                work_dir=tmpdir,
                use_docker=False,
            ) as sandbox:
                assert sandbox._running == True
            
            assert sandbox._running == False


class TestSecureSandboxManager:
    """Test secure sandbox manager"""
    
    def test_manager_create_and_list(self):
        """Test manager create and list operations"""
        from kas.core.security import SecureSandboxManager
        
        manager = SecureSandboxManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SecureSandboxConfig(
                name="sandbox1",
                work_dir=Path(tmpdir) / "sb1",
                use_docker=False,
            )
            sandbox = manager.create(config)
            
            assert "sandbox1" in manager.list()
            assert manager.get("sandbox1") == sandbox
    
    def test_manager_remove(self):
        """Test manager remove operation"""
        from kas.core.security import SecureSandboxManager
        
        manager = SecureSandboxManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SecureSandboxConfig(
                name="sandbox2",
                work_dir=Path(tmpdir) / "sb2",
                use_docker=False,
            )
            manager.create(config)
            
            assert manager.remove("sandbox2") == True
            assert "sandbox2" not in manager.list()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
