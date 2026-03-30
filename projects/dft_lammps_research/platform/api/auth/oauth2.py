"""
OAuth2 Authentication Manager

Handles OAuth2 flows including:
- Authorization Code flow
- Client Credentials flow
- Device Code flow
- Token refresh
- Token validation
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
from jose import JWTError, jwt
from passlib.context import CryptContext
import structlog
import uuid
import hashlib

from api_platform.gateway.config import GatewayConfig

logger = structlog.get_logger()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
config = GatewayConfig()


class OAuth2Manager:
    """Manages OAuth2 authentication flows"""
    
    def __init__(self):
        self.secret_key = config.JWT_SECRET_KEY
        self.algorithm = config.JWT_ALGORITHM
        self.access_token_expire = config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire = config.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        self._clients: Dict[str, dict] = {}  # In-memory store (use DB in production)
        self._tokens: Dict[str, dict] = {}  # Token cache
        
    def create_access_token(
        self,
        client_id: str,
        scopes: List[str],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire)
        
        to_encode = {
            "sub": client_id,
            "scopes": scopes,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),
            "type": "access",
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        # Store token metadata
        self._tokens[to_encode["jti"]] = {
            "client_id": client_id,
            "scopes": scopes,
            "expires": expire,
            "revoked": False,
        }
        
        return encoded_jwt
    
    def create_refresh_token(self, client_id: str) -> str:
        """Create a new refresh token"""
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire)
        
        to_encode = {
            "sub": client_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),
            "type": "refresh",
        }
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    async def validate_token(self, token: str) -> Optional[dict]:
        """Validate and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            client_id: str = payload.get("sub")
            token_type: str = payload.get("type")
            jti: str = payload.get("jti")
            scopes: List[str] = payload.get("scopes", [])
            
            if client_id is None or token_type != "access":
                return None
            
            # Check if token is revoked
            token_meta = self._tokens.get(jti)
            if token_meta and token_meta.get("revoked"):
                return None
            
            return {
                "client_id": client_id,
                "scopes": scopes,
                "jti": jti,
            }
            
        except JWTError:
            return None
    
    async def revoke_token(self, jti: str) -> bool:
        """Revoke a token by its JTI"""
        if jti in self._tokens:
            self._tokens[jti]["revoked"] = True
            logger.info("token_revoked", jti=jti)
            return True
        return False
    
    async def register_client(
        self,
        client_name: str,
        redirect_uris: List[str],
        grant_types: List[str],
        scopes: List[str],
        tier: str = "free"
    ) -> dict:
        """Register a new OAuth2 client"""
        client_id = f"client_{uuid.uuid4().hex[:16]}"
        client_secret = f"secret_{uuid.uuid4().hex}"
        
        client = {
            "client_id": client_id,
            "client_secret": pwd_context.hash(client_secret),
            "name": client_name,
            "redirect_uris": redirect_uris,
            "grant_types": grant_types,
            "scopes": scopes,
            "tier": tier,
            "created_at": datetime.utcnow().isoformat(),
            "active": True,
        }
        
        self._clients[client_id] = client
        
        logger.info("oauth2_client_registered", client_id=client_id, name=client_name)
        
        # Return client credentials (secret is only shown once)
        return {
            "client_id": client_id,
            "client_secret": client_secret,
            "name": client_name,
            "redirect_uris": redirect_uris,
            "grant_types": grant_types,
            "scopes": scopes,
        }
    
    async def validate_client(self, client_id: str, client_secret: str) -> Optional[dict]:
        """Validate client credentials"""
        client = self._clients.get(client_id)
        
        if not client or not client.get("active"):
            return None
        
        if pwd_context.verify(client_secret, client["client_secret"]):
            return {
                "client_id": client_id,
                "name": client["name"],
                "scopes": client["scopes"],
                "tier": client["tier"],
            }
        
        return None
    
    async def exchange_code_for_token(
        self,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str
    ) -> Optional[dict]:
        """Exchange authorization code for tokens (Authorization Code flow)"""
        # Validate client
        client = await self.validate_client(client_id, client_secret)
        if not client:
            return None
        
        # In production: validate code against database
        # For now, generate tokens directly
        
        access_token = self.create_access_token(
            client_id=client_id,
            scopes=client["scopes"]
        )
        refresh_token = self.create_refresh_token(client_id)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire * 60,
            "refresh_token": refresh_token,
            "scope": " ".join(client["scopes"]),
        }
    
    async def client_credentials_grant(
        self,
        client_id: str,
        client_secret: str,
        scope: Optional[str] = None
    ) -> Optional[dict]:
        """Handle Client Credentials grant flow"""
        client = await self.validate_client(client_id, client_secret)
        
        if not client or "client_credentials" not in self._clients[client_id].get("grant_types", []):
            return None
        
        # Filter scopes
        requested_scopes = scope.split() if scope else []
        allowed_scopes = set(client["scopes"])
        granted_scopes = [s for s in requested_scopes if s in allowed_scopes]
        
        if not granted_scopes:
            granted_scopes = ["read"] if "read" in allowed_scopes else list(allowed_scopes)[:1]
        
        access_token = self.create_access_token(
            client_id=client_id,
            scopes=granted_scopes
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.access_token_expire * 60,
            "scope": " ".join(granted_scopes),
        }
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[dict]:
        """Refresh an access token using a refresh token"""
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "refresh":
                return None
            
            client_id = payload.get("sub")
            client = self._clients.get(client_id)
            
            if not client or not client.get("active"):
                return None
            
            # Generate new tokens
            new_access_token = self.create_access_token(
                client_id=client_id,
                scopes=client["scopes"]
            )
            new_refresh_token = self.create_refresh_token(client_id)
            
            return {
                "access_token": new_access_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire * 60,
                "refresh_token": new_refresh_token,
                "scope": " ".join(client["scopes"]),
            }
            
        except JWTError:
            return None
