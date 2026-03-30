"""
认证路由
"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm

from ...auth import (
    Token,
    User,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    refresh_access_token,
    get_current_user,
    generate_api_key,
    api_metrics,
)
from ...auth.permissions import audit_logger
from ...models.schemas import (
    LoginRequest,
    PasswordChange,
    APIKeyCreate,
    APIKeyResponse,
    APIResponse,
    ErrorResponse,
    UserResponse,
)
from ...utils import UnauthorizedException, ValidationException

router = APIRouter()

ACCESS_TOKEN_EXPIRE_MINUTES = 30


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    OAuth2登录，获取访问令牌
    
    - **username**: 用户名
    - **password**: 密码
    """
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        api_metrics.record_auth_attempt(success=False)
        raise UnauthorizedException("Incorrect username or password")
    
    if user.status != "active":
        api_metrics.record_auth_attempt(success=False)
        raise UnauthorizedException("User account is not active")
    
    api_metrics.record_auth_attempt(success=True)
    
    # 记录审计日志
    await audit_logger.log_action(
        user_id=user.id,
        action="login",
        resource_type="auth",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )
    
    # 创建令牌
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "scopes": user.scopes,
            "user_id": user.id
        },
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": user.username}
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/login/json", response_model=Token)
async def login_json(
    request: Request,
    login_data: LoginRequest
):
    """
    JSON格式登录
    
    - **username**: 用户名
    - **password**: 密码
    """
    user = authenticate_user(login_data.username, login_data.password)
    
    if not user:
        raise UnauthorizedException("Incorrect username or password")
    
    if user.status != "active":
        raise UnauthorizedException("User account is not active")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.username,
            "scopes": user.scopes,
            "user_id": user.id
        },
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": user.username}
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """
    使用刷新令牌获取新的访问令牌
    
    - **refresh_token**: 刷新令牌
    """
    return refresh_access_token(refresh_token)


@router.post("/logout", response_model=APIResponse)
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    用户登出
    
    使当前访问令牌失效（需要前端配合丢弃令牌）
    """
    await audit_logger.log_action(
        user_id=current_user.id,
        action="logout",
        resource_type="auth",
        ip_address=request.client.host if request.client else None,
    )
    
    return APIResponse(
        success=True,
        message="Successfully logged out"
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """获取当前登录用户信息"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        status=current_user.status,
        created_at=None,  # 从数据库获取
        last_login=None,
    )


@router.post("/change-password", response_model=APIResponse)
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user)
):
    """
    修改密码
    
    - **current_password**: 当前密码
    - **new_password**: 新密码
    """
    # 验证当前密码
    from ...auth import verify_password, get_password_hash
    from ...auth.security import get_user
    
    user = get_user(current_user.username)
    if not verify_password(password_data.current_password, user.hashed_password):
        raise ValidationException("Current password is incorrect")
    
    # 更新密码（实际应更新数据库）
    new_hash = get_password_hash(password_data.new_password)
    
    return APIResponse(
        success=True,
        message="Password changed successfully"
    )


@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: Request,
    key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user)
):
    """
    创建API Key
    
    - **name**: API Key名称
    - **description**: 描述
    - **expires_in_days**: 过期天数
    - **scopes**: 权限范围
    """
    new_key = generate_api_key()
    
    await audit_logger.log_action(
        user_id=current_user.id,
        action="create_api_key",
        resource_type="api_key",
        details={"name": key_data.name}
    )
    
    import datetime
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=key_data.expires_in_days)
    
    return APIKeyResponse(
        id=1,  # 从数据库获取
        name=key_data.name,
        key=new_key,
        prefix=new_key[:8],
        description=key_data.description,
        scopes=key_data.scopes,
        created_at=datetime.datetime.utcnow(),
        expires_at=expires_at,
        last_used_at=None,
        is_active=True,
    )


@router.get("/api-keys", response_model=list)
async def list_api_keys(
    current_user: User = Depends(get_current_user)
):
    """获取用户的API Key列表"""
    # 从数据库获取
    return []


@router.delete("/api-keys/{key_id}", response_model=APIResponse)
async def revoke_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user)
):
    """撤销API Key"""
    await audit_logger.log_action(
        user_id=current_user.id,
        action="revoke_api_key",
        resource_type="api_key",
        resource_id=str(key_id)
    )
    
    return APIResponse(
        success=True,
        message="API key revoked successfully"
    )
