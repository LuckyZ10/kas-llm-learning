"""
File Management API Endpoints
"""
import os
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, StreamingResponse

from app.models.user import User
from app.services.auth import get_current_user
from app.core.config import settings

router = APIRouter()


@router.get("/list")
async def list_files(
    path: str = Query(".", description="Relative path from work directory"),
    pattern: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    """List files in directory"""
    base_path = settings.WORK_DIR / path
    
    # Security check
    try:
        base_path.resolve().relative_to(settings.WORK_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not base_path.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    
    files = []
    search_pattern = pattern or "*"
    
    for item in base_path.glob(search_pattern):
        stat = item.stat()
        files.append({
            "name": item.name,
            "path": str(item.relative_to(settings.WORK_DIR)),
            "type": "directory" if item.is_dir() else "file",
            "size": stat.st_size if item.is_file() else None,
            "modified": stat.st_mtime,
        })
    
    return {"path": path, "files": files}


@router.get("/download/{file_path:path}")
async def download_file(
    file_path: str,
    current_user: User = Depends(get_current_user),
):
    """Download a file"""
    full_path = settings.WORK_DIR / file_path
    
    # Security check
    try:
        full_path.resolve().relative_to(settings.WORK_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=full_path,
        filename=full_path.name,
        media_type="application/octet-stream",
    )


@router.post("/upload")
async def upload_file(
    path: str = Query(".", description="Target directory"),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """Upload a file"""
    target_dir = settings.UPLOAD_DIR / path
    
    # Security check
    try:
        target_dir.resolve().relative_to(settings.UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = target_dir / file.filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return {
        "filename": file.filename,
        "path": str(file_path.relative_to(settings.WORK_DIR)),
        "size": len(content),
    }


@router.get("/structure/{structure_id}/view")
async def view_structure(
    structure_id: str,
    format: str = Query("cif", enum=["cif", "poscar", "xyz", "json"]),
    current_user: User = Depends(get_current_user),
):
    """Get structure file for visualization"""
    # This would integrate with ASE/Pymatgen
    # For now return placeholder
    return {"structure_id": structure_id, "format": format}
