"""
用户管理路由
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query

from ...auth import (
    User,
    get_current_active_user,
    Permission,
    has_permission,
)
from ...models.schemas import (
    UserCreate,
    UserUpdate,
    UserResponse,
    APIResponse,
    PaginatedResponse,
    UserRole,
    UserStatus,
)
from ...utils import ForbiddenException, NotFoundException

router = APIRouter()


@router.get("", response_model=PaginatedResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    role: Optional[UserRole] = None,
    status: Optional[UserStatus] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    获取用户列表
    
    需要管理员权限
    """
    if not has_permission(current_user.role, Permission.USER_READ):
        raise ForbiddenException("Admin permission required")
    
    # 模拟用户数据
    users = [
        UserResponse(
            id=1,
            username="admin",
            email="admin@dftlammps.org",
            full_name="Administrator",
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            created_at=None,
            last_login=None,
        ),
        UserResponse(
            id=2,
            username="demo",
            email="demo@dftlammps.org",
            full_name="Demo User",
            role=UserRole.RESEARCHER,
            status=UserStatus.ACTIVE,
            created_at=None,
            last_login=None,
        ),
    ]
    
    # 过滤
    if role:
        users = [u for u in users if u.role == role]
    if status:
        users = [u for u in users if u.status == status]
    if search:
        users = [u for u in users if search.lower() in u.username.lower()]
    
    # 分页
    total = len(users)
    start = (page - 1) * page_size
    end = start + page_size
    
    return PaginatedResponse(
        success=True,
        data=users[start:end],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
        has_next=end < total,
        has_prev=page > 1,
    )


@router.post("", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(get_current_active_user)
):
    """
    创建新用户
    
    需要管理员权限
    """
    if not has_permission(current_user.role, Permission.USER_CREATE):
        raise ForbiddenException("Admin permission required")
    
    # 实际应写入数据库
    return UserResponse(
        id=3,
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name,
        role=user_data.role,
        status=UserStatus.PENDING,
        created_at=None,
        last_login=None,
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """获取用户详情"""
    if current_user.id != user_id and not has_permission(current_user.role, Permission.USER_READ):
        raise ForbiddenException("Permission denied")
    
    # 模拟查询
    if user_id == 1:
        return UserResponse(
            id=1,
            username="admin",
            email="admin@dftlammps.org",
            full_name="Administrator",
            role=UserRole.ADMIN,
            status=UserStatus.ACTIVE,
            created_at=None,
            last_login=None,
        )
    
    raise NotFoundException("User", str(user_id))


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """更新用户信息"""
    if current_user.id != user_id and not has_permission(current_user.role, Permission.USER_UPDATE):
        raise ForbiddenException("Permission denied")
    
    return UserResponse(
        id=user_id,
        username="updated_user",
        email=user_data.email or "updated@example.com",
        full_name=user_data.full_name,
        role=user_data.role or UserRole.RESEARCHER,
        status=user_data.status or UserStatus.ACTIVE,
        created_at=None,
        last_login=None,
    )


@router.delete("/{user_id}", response_model=APIResponse)
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """删除用户"""
    if not has_permission(current_user.role, Permission.USER_DELETE):
        raise ForbiddenException("Admin permission required")
    
    if user_id == current_user.id:
        raise ForbiddenException("Cannot delete yourself")
    
    return APIResponse(
        success=True,
        message=f"User {user_id} deleted successfully"
    )


@router.get("/{user_id}/quota")
async def get_user_quota(
    user_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """获取用户配额使用情况"""
    if current_user.id != user_id and not has_permission(current_user.role, Permission.USER_READ):
        raise ForbiddenException("Permission denied")
    
    return {
        "user_id": user_id,
        "quota_limit": current_user.quota_limit,
        "quota_used": current_user.quota_used,
        "quota_remaining": current_user.quota_limit - current_user.quota_used,
        "reset_date": None,  # 下个月初
    }
