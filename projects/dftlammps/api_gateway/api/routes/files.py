"""
文件管理路由
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, UploadFile, File, Form, Request

from ...auth import (
    User,
    get_current_user_or_api_key,
    Permission,
    has_permission,
)
from ...models.schemas import FileUploadResponse, FileInfo, APIResponse
from ...utils import ForbiddenException, NotFoundException, humanize_bytes
from ...utils.helpers import generate_short_id

router = APIRouter()

# 模拟文件存储
FILES_DB = {}


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    上传文件
    
    支持POSCAR、CIF、XYZ等格式
    最大文件大小：100MB
    """
    if not has_permission(current_user.role, Permission.WRITE):
        raise ForbiddenException("Permission denied")
    
    import datetime
    
    file_id = generate_short_id()
    
    # 读取文件内容
    content = await file.read()
    size = len(content)
    
    # 存储文件信息
    file_info = {
        "file_id": file_id,
        "filename": file.filename,
        "size": size,
        "content_type": file.content_type,
        "uploaded_at": datetime.datetime.utcnow(),
        "uploaded_by": current_user.id,
        "description": description,
    }
    
    FILES_DB[file_id] = file_info
    
    return FileUploadResponse(
        file_id=file_id,
        filename=file.filename,
        size=size,
        content_type=file.content_type,
        uploaded_at=file_info["uploaded_at"],
        url=f"/api/v1/files/{file_id}",
    )


@router.get("", response_model=List[FileInfo])
async def list_files(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取文件列表"""
    # 获取用户的文件
    user_files = [
        f for f in FILES_DB.values()
        if f.get("uploaded_by") == current_user.id
    ]
    
    return [
        FileInfo(
            file_id=f["file_id"],
            filename=f["filename"],
            size=f["size"],
            content_type=f["content_type"],
            uploaded_at=f["uploaded_at"],
            uploaded_by=f["uploaded_by"],
            metadata={"description": f.get("description")},
        )
        for f in user_files
    ]


@router.get("/{file_id}")
async def get_file(
    file_id: str,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取文件详情"""
    file_info = FILES_DB.get(file_id)
    
    if not file_info:
        raise NotFoundException("File", file_id)
    
    if file_info["uploaded_by"] != current_user.id and not has_permission(current_user.role, Permission.ADMIN):
        raise ForbiddenException("Access denied")
    
    return FileInfo(
        file_id=file_info["file_id"],
        filename=file_info["filename"],
        size=file_info["size"],
        content_type=file_info["content_type"],
        uploaded_at=file_info["uploaded_at"],
        uploaded_by=file_info["uploaded_by"],
    )


@router.get("/{file_id}/download")
async def download_file(
    file_id: str,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """下载文件"""
    from fastapi.responses import StreamingResponse
    import io
    
    file_info = FILES_DB.get(file_id)
    
    if not file_info:
        raise NotFoundException("File", file_id)
    
    if file_info["uploaded_by"] != current_user.id and not has_permission(current_user.role, Permission.ADMIN):
        raise ForbiddenException("Access denied")
    
    # 模拟文件内容
    content = f"# File content for {file_info['filename']}\n"
    
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type=file_info["content_type"],
        headers={
            "Content-Disposition": f"attachment; filename={file_info['filename']}"
        }
    )


@router.delete("/{file_id}", response_model=APIResponse)
async def delete_file(
    file_id: str,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """删除文件"""
    file_info = FILES_DB.get(file_id)
    
    if not file_info:
        raise NotFoundException("File", file_id)
    
    if file_info["uploaded_by"] != current_user.id and not has_permission(current_user.role, Permission.ADMIN):
        raise ForbiddenException("Access denied")
    
    del FILES_DB[file_id]
    
    return APIResponse(
        success=True,
        message=f"File {file_id} deleted"
    )
