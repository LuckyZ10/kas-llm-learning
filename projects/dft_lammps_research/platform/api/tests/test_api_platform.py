"""
Tests for API Platform

Run with: pytest tests/ -v
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from api_platform.gateway.config import GatewayConfig
from api_platform.auth.oauth2 import OAuth2Manager
from api_platform.auth.api_key import APIKeyManager
from api_platform.auth.permissions import PermissionChecker, Permission, Role
from api_platform.webhooks.manager import WebhookManager, WebhookEventType


# ============== Authentication Tests ==============

class TestOAuth2Manager:
    def setup_method(self):
        self.manager = OAuth2Manager()
    
    def test_create_access_token(self):
        token = self.manager.create_access_token(
            client_id="client_123",
            scopes=["read", "write"]
        )
        assert token is not None
        assert isinstance(token, str)
    
    def test_create_refresh_token(self):
        token = self.manager.create_refresh_token("client_123")
        assert token is not None
        assert isinstance(token, str)
    
    @pytest.mark.asyncio
    async def test_validate_token_valid(self):
        token = self.manager.create_access_token(
            client_id="client_123",
            scopes=["read"]
        )
        result = await self.manager.validate_token(token)
        assert result is not None
        assert result["client_id"] == "client_123"
        assert "read" in result["scopes"]
    
    @pytest.mark.asyncio
    async def test_validate_token_invalid(self):
        result = await self.manager.validate_token("invalid.token.here")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_register_client(self):
        client = await self.manager.register_client(
            client_name="Test App",
            redirect_uris=["http://localhost/callback"],
            grant_types=["client_credentials"],
            scopes=["read", "write"],
            tier="pro"
        )
        assert client["client_id"].startswith("client_")
        assert client["client_secret"].startswith("secret_")
        assert client["name"] == "Test App"


class TestAPIKeyManager:
    def setup_method(self):
        self.manager = APIKeyManager()
    
    def test_generate_api_key(self):
        key, key_id = self.manager.generate_api_key(
            client_id="client_123",
            name="Test Key",
            tier="pro"
        )
        assert key.startswith("dftl_")
        assert key_id.startswith("key_")
        assert len(key) > 32
    
    @pytest.mark.asyncio
    async def test_validate_key_valid(self):
        key, key_id = self.manager.generate_api_key(
            client_id="client_123",
            name="Test Key"
        )
        result = await self.manager.validate_key(key)
        assert result is not None
        assert result["client_id"] == "client_123"
        assert result["tier"] == "free"
    
    @pytest.mark.asyncio
    async def test_validate_key_invalid(self):
        result = await self.manager.validate_key("invalid_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_validate_key_wrong_prefix(self):
        result = await self.manager.validate_key("not_dftl_prefix")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_revoke_key(self):
        key, key_id = self.manager.generate_api_key(
            client_id="client_123",
            name="Test Key"
        )
        success = await self.manager.revoke_key(key_id, "client_123")
        assert success is True
        
        # Verify revoked
        result = await self.manager.validate_key(key)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_client_keys(self):
        self.manager.generate_api_key("client_123", "Key 1")
        self.manager.generate_api_key("client_123", "Key 2")
        
        keys = await self.manager.get_client_keys("client_123")
        assert len(keys) == 2


class TestPermissionChecker:
    def setup_method(self):
        self.checker = PermissionChecker()
    
    def test_get_role_permissions(self):
        perms = self.checker.get_role_permissions(Role.ADMIN)
        assert Permission.READ_PROJECTS in perms
        assert Permission.DELETE_RESOURCES in perms
    
    def test_has_permission_true(self):
        result = self.checker.has_permission(
            ["projects:read", "projects:write"],
            Permission.READ_PROJECTS
        )
        assert result is True
    
    def test_has_permission_false(self):
        result = self.checker.has_permission(
            ["projects:read"],
            Permission.DELETE_RESOURCES
        )
        assert result is False
    
    def test_has_permission_wildcard(self):
        result = self.checker.has_permission(
            ["*"],
            Permission.DELETE_RESOURCES
        )
        assert result is True
    
    def test_get_tier_limits_free(self):
        limits = self.checker.get_tier_limits("free")
        assert limits["max_projects"] == 5
        assert limits["rate_limit_per_minute"] == 60
    
    def test_get_tier_limits_enterprise(self):
        limits = self.checker.get_tier_limits("enterprise")
        assert limits["max_projects"] == -1  # unlimited


# ============== Webhook Tests ==============

class TestWebhookManager:
    def setup_method(self):
        self.manager = WebhookManager()
    
    @pytest.mark.asyncio
    async def test_subscribe(self):
        result = await self.manager.subscribe(
            client_id="client_123",
            url="https://example.com/webhook",
            events=["calculation.completed"],
            metadata={"team": "research"}
        )
        assert result["webhook_id"].startswith("wh_")
        assert result["secret"].startswith("whsec_")
        assert result["url"] == "https://example.com/webhook"
    
    def test_create_signature(self):
        payload = '{"event": "test"}'
        secret = "whsec_test_secret"
        signature = self.manager.create_signature(payload, secret)
        assert signature.startswith("sha256=")
    
    def test_verify_signature_valid(self):
        payload = '{"event": "test"}'
        secret = "whsec_test_secret"
        signature = self.manager.create_signature(payload, secret)
        result = self.manager.verify_signature(payload, signature, secret)
        assert result is True
    
    def test_verify_signature_invalid(self):
        payload = '{"event": "test"}'
        result = self.manager.verify_signature(
            payload,
            "sha256=wrong_signature",
            "whsec_test"
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_subscriptions(self):
        await self.manager.subscribe(
            client_id="client_123",
            url="https://example.com/webhook1",
            events=["calculation.completed"]
        )
        await self.manager.subscribe(
            client_id="client_123",
            url="https://example.com/webhook2",
            events=["project.completed"]
        )
        
        subs = await self.manager.get_subscriptions("client_123")
        assert len(subs) == 2
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        sub = await self.manager.subscribe(
            client_id="client_123",
            url="https://example.com/webhook",
            events=["calculation.completed"]
        )
        
        success = await self.manager.unsubscribe(sub["webhook_id"], "client_123")
        assert success is True
        
        subs = await self.manager.get_subscriptions("client_123")
        assert len(subs) == 0


# ============== Integration Tests ==============

@pytest.mark.asyncio
async def test_full_api_flow():
    """Test a complete API flow"""
    # This would be a full integration test
    # For now, just verify components work together
    
    oauth = OAuth2Manager()
    api_keys = APIKeyManager()
    webhooks = WebhookManager()
    
    # Create API key
    key, key_id = api_keys.generate_api_key("client_123", "Test Key")
    
    # Validate key
    auth_result = await api_keys.validate_key(key)
    assert auth_result is not None
    
    # Subscribe webhook
    sub = await webhooks.subscribe(
        client_id="client_123",
        url="https://example.com/webhook",
        events=["calculation.completed"]
    )
    assert sub["webhook_id"] is not None


# ============== Configuration Tests ==============

class TestGatewayConfig:
    def test_default_config(self):
        config = GatewayConfig()
        assert config.VERSION == "1.0.0"
        assert config.PORT == 8080
        assert "free" in config.RATE_LIMITS
    
    def test_rate_limits_structure(self):
        config = GatewayConfig()
        free_limits = config.RATE_LIMITS["free"]
        assert "requests_per_minute" in free_limits
        assert "requests_per_day" in free_limits
        assert "max_projects" in free_limits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
