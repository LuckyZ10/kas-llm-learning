"""
API Key Authentication Manager

Manages API key lifecycle:
- Key generation
- Key validation
- Key rotation
- Key revocation
- Usage tracking
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
import secrets
import hashlib
import structlog
import uuid

logger = structlog.get_logger()


class APIKeyManager:
    """Manages API key authentication"""
    
    def __init__(self):
        self._keys: Dict[str, dict] = {}  # hashed_key -> key_data
        self._clients: Dict[str, List[str]] = {}  # client_id -> [hashed_keys]
    
    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def generate_api_key(
        self,
        client_id: str,
        name: str,
        tier: str = "free",
        permissions: List[str] = None,
        expires_days: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> tuple[str, str]:
        """
        Generate a new API key
        
        Returns:
            Tuple of (api_key, key_id) - only time the raw key is visible
        """
        # Generate secure random key
        raw_key = f"dftl_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(raw_key)
        key_id = f"key_{uuid.uuid4().hex[:12]}"
        
        # Calculate expiration
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        key_data = {
            "key_id": key_id,
            "key_hash": key_hash,
            "client_id": client_id,
            "name": name,
            "tier": tier,
            "permissions": permissions or ["read"],
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "last_used_at": None,
            "usage_count": 0,
            "active": True,
            "metadata": metadata or {},
        }
        
        # Store key
        self._keys[key_hash] = key_data
        
        # Track client's keys
        if client_id not in self._clients:
            self._clients[client_id] = []
        self._clients[client_id].append(key_hash)
        
        logger.info(
            "api_key_generated",
            key_id=key_id,
            client_id=client_id,
            tier=tier,
            expires_at=expires_at.isoformat() if expires_at else None
        )
        
        return raw_key, key_id
    
    async def validate_key(self, api_key: str) -> Optional[dict]:
        """Validate an API key"""
        if not api_key or not api_key.startswith("dftl_"):
            return None
        
        key_hash = self._hash_key(api_key)
        key_data = self._keys.get(key_hash)
        
        if not key_data:
            return None
        
        if not key_data.get("active"):
            logger.warning("api_key_inactive", key_id=key_data.get("key_id"))
            return None
        
        # Check expiration
        if key_data.get("expires_at"):
            if datetime.utcnow() > key_data["expires_at"]:
                logger.warning("api_key_expired", key_id=key_data.get("key_id"))
                return None
        
        # Update usage stats
        key_data["last_used_at"] = datetime.utcnow()
        key_data["usage_count"] += 1
        
        return {
            "key_id": key_data["key_id"],
            "client_id": key_data["client_id"],
            "name": key_data["name"],
            "tier": key_data["tier"],
            "permissions": key_data["permissions"],
            "metadata": key_data["metadata"],
        }
    
    async def revoke_key(self, key_id: str, client_id: str) -> bool:
        """Revoke an API key"""
        # Find key by ID
        for key_hash, key_data in self._keys.items():
            if key_data["key_id"] == key_id and key_data["client_id"] == client_id:
                key_data["active"] = False
                logger.info("api_key_revoked", key_id=key_id, client_id=client_id)
                return True
        return False
    
    async def rotate_key(
        self,
        old_key_id: str,
        client_id: str,
        grace_period_days: int = 7
    ) -> Optional[tuple[str, str]]:
        """
        Rotate an API key (generate new, mark old for deletion)
        
        Args:
            old_key_id: The key ID to rotate
            client_id: Owner of the key
            grace_period_days: Days before old key is fully revoked
        
        Returns:
            Tuple of (new_api_key, new_key_id) or None if failed
        """
        # Find old key
        old_key_data = None
        old_key_hash = None
        
        for key_hash, key_data in self._keys.items():
            if key_data["key_id"] == old_key_id and key_data["client_id"] == client_id:
                old_key_data = key_data
                old_key_hash = key_hash
                break
        
        if not old_key_data:
            return None
        
        # Generate new key with same permissions
        new_key, new_key_id = self.generate_api_key(
            client_id=client_id,
            name=f"{old_key_data['name']} (rotated)",
            tier=old_key_data["tier"],
            permissions=old_key_data["permissions"],
            expires_days=old_key_data["expires_at"].days if old_key_data["expires_at"] else None,
            metadata=old_key_data["metadata"]
        )
        
        # Set old key to expire after grace period
        old_key_data["expires_at"] = datetime.utcnow() + timedelta(days=grace_period_days)
        old_key_data["rotated_to"] = new_key_id
        
        logger.info(
            "api_key_rotated",
            old_key_id=old_key_id,
            new_key_id=new_key_id,
            client_id=client_id,
            grace_period_days=grace_period_days
        )
        
        return new_key, new_key_id
    
    async def get_client_keys(self, client_id: str) -> List[dict]:
        """Get all API keys for a client (without sensitive data)"""
        keys = []
        for key_hash in self._clients.get(client_id, []):
            key_data = self._keys.get(key_hash)
            if key_data:
                keys.append({
                    "key_id": key_data["key_id"],
                    "name": key_data["name"],
                    "tier": key_data["tier"],
                    "permissions": key_data["permissions"],
                    "created_at": key_data["created_at"].isoformat(),
                    "expires_at": key_data["expires_at"].isoformat() if key_data["expires_at"] else None,
                    "last_used_at": key_data["last_used_at"].isoformat() if key_data["last_used_at"] else None,
                    "usage_count": key_data["usage_count"],
                    "active": key_data["active"],
                })
        return keys
    
    async def get_usage_stats(self, key_id: str, client_id: str) -> Optional[dict]:
        """Get usage statistics for a specific key"""
        for key_hash, key_data in self._keys.items():
            if key_data["key_id"] == key_id and key_data["client_id"] == client_id:
                return {
                    "key_id": key_id,
                    "usage_count": key_data["usage_count"],
                    "last_used_at": key_data["last_used_at"].isoformat() if key_data["last_used_at"] else None,
                    "created_at": key_data["created_at"].isoformat(),
                    "active": key_data["active"],
                }
        return None
    
    async def cleanup_expired_keys(self) -> int:
        """Clean up expired and revoked keys. Returns count of removed keys."""
        now = datetime.utcnow()
        removed = 0
        
        keys_to_remove = []
        for key_hash, key_data in self._keys.items():
            # Remove if expired and grace period passed (7 days after expiry)
            if key_data.get("expires_at"):
                grace_end = key_data["expires_at"] + timedelta(days=7)
                if now > grace_end:
                    keys_to_remove.append(key_hash)
                    removed += 1
        
        for key_hash in keys_to_remove:
            del self._keys[key_hash]
        
        if removed > 0:
            logger.info("expired_keys_cleaned", count=removed)
        
        return removed
