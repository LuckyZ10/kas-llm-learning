"""
Data pipeline for file transfer and synchronization.
"""

import os
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..core.base import BaseHPCConnector
from ..core.exceptions import DataTransferError

logger = logging.getLogger(__name__)


@dataclass
class TransferProgress:
    """File transfer progress."""
    source: str
    destination: str
    bytes_transferred: int
    total_bytes: int
    percentage: float
    speed_mbps: float
    eta_seconds: Optional[float] = None
    
    @property
    def is_complete(self) -> bool:
        return self.bytes_transferred >= self.total_bytes


@dataclass
class SyncManifest:
    """Synchronization manifest."""
    files: Dict[str, Dict[str, any]] = field(default_factory=dict)
    
    def add_file(self, path: str, size: int, mtime: float, checksum: str = None):
        """Add a file to the manifest."""
        self.files[path] = {
            'size': size,
            'mtime': mtime,
            'checksum': checksum,
        }
    
    def to_dict(self) -> Dict:
        return {'files': self.files}
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SyncManifest":
        manifest = cls()
        manifest.files = data.get('files', {})
        return manifest


class DataPipeline:
    """Data pipeline for HPC data transfers."""
    
    def __init__(self, connector: BaseHPCConnector, staging_dir: str = None):
        self.connector = connector
        self.staging_dir = staging_dir or "~/hpc_staging"
        self._transfer_semaphore = asyncio.Semaphore(5)  # Limit concurrent transfers
    
    async def upload_files(
        self,
        files: List[Tuple[str, str]],
        progress_callback: Optional[Callable[[TransferProgress], None]] = None,
        verify_checksums: bool = False
    ) -> Dict[str, bool]:
        """
        Upload multiple files.
        
        Args:
            files: List of (local_path, remote_path) tuples
            progress_callback: Progress callback
            verify_checksums: Whether to verify checksums after transfer
            
        Returns:
            Dictionary mapping local_path to success status
        """
        results = {}
        
        async def upload_single(local_path: str, remote_path: str) -> bool:
            try:
                async with self._transfer_semaphore:
                    await self._upload_with_progress(
                        local_path, remote_path, progress_callback
                    )
                    
                    if verify_checksums:
                        local_checksum = await self._calculate_local_checksum(local_path)
                        remote_checksum = await self._calculate_remote_checksum(remote_path)
                        if local_checksum != remote_checksum:
                            raise DataTransferError(
                                f"Checksum mismatch for {local_path}"
                            )
                    
                    return True
            except Exception as e:
                logger.error(f"Failed to upload {local_path}: {e}")
                return False
        
        # Execute uploads concurrently
        tasks = [
            upload_single(local, remote)
            for local, remote in files
        ]
        
        completed = await asyncio.gather(*tasks)
        
        for (local_path, _), success in zip(files, completed):
            results[local_path] = success
        
        return results
    
    async def download_files(
        self,
        files: List[Tuple[str, str]],
        progress_callback: Optional[Callable[[TransferProgress], None]] = None
    ) -> Dict[str, bool]:
        """
        Download multiple files.
        
        Args:
            files: List of (remote_path, local_path) tuples
            progress_callback: Progress callback
            
        Returns:
            Dictionary mapping remote_path to success status
        """
        results = {}
        
        async def download_single(remote_path: str, local_path: str) -> bool:
            try:
                async with self._transfer_semaphore:
                    await self._download_with_progress(
                        remote_path, local_path, progress_callback
                    )
                    return True
            except Exception as e:
                logger.error(f"Failed to download {remote_path}: {e}")
                return False
        
        tasks = [
            download_single(remote, local)
            for remote, local in files
        ]
        
        completed = await asyncio.gather(*tasks)
        
        for (remote_path, _), success in zip(files, completed):
            results[remote_path] = success
        
        return results
    
    async def sync_to_remote(
        self,
        local_dir: str,
        remote_dir: str,
        exclude_patterns: List[str] = None,
        delete_remote: bool = False,
        progress_callback: Optional[Callable[[TransferProgress], None]] = None
    ) -> SyncManifest:
        """
        Synchronize local directory to remote.
        
        Args:
            local_dir: Local directory path
            remote_dir: Remote directory path
            exclude_patterns: Patterns to exclude
            delete_remote: Delete remote files not present locally
            progress_callback: Progress callback
            
        Returns:
            SyncManifest of synchronized files
        """
        exclude_patterns = exclude_patterns or []
        
        # Build local manifest
        local_manifest = await self._build_local_manifest(local_dir, exclude_patterns)
        
        # Build remote manifest
        remote_manifest = await self._build_remote_manifest(remote_dir)
        
        # Determine files to transfer
        files_to_upload = []
        for rel_path, local_info in local_manifest.files.items():
            if rel_path not in remote_manifest.files:
                # New file
                files_to_upload.append(rel_path)
            else:
                # Compare mtime and size
                remote_info = remote_manifest.files[rel_path]
                if (local_info['size'] != remote_info['size'] or
                    local_info['mtime'] > remote_info['mtime']):
                    files_to_upload.append(rel_path)
        
        # Transfer files
        total_size = sum(
            local_manifest.files[p]['size'] for p in files_to_upload
        )
        transferred = 0
        
        async def progress_wrapper(bytes_done: int, total: int):
            if progress_callback:
                progress_callback(TransferProgress(
                    source="sync",
                    destination=remote_dir,
                    bytes_transferred=transferred + bytes_done,
                    total_bytes=total_size,
                    percentage=(transferred + bytes_done) / total_size * 100 if total_size > 0 else 100,
                    speed_mbps=0,
                ))
        
        for rel_path in files_to_upload:
            local_path = os.path.join(local_dir, rel_path)
            remote_path = f"{remote_dir}/{rel_path}"
            
            await self.connector.upload_file(
                local_path, remote_path, progress_wrapper
            )
            transferred += local_manifest.files[rel_path]['size']
        
        # Handle deletions if requested
        if delete_remote:
            for rel_path in remote_manifest.files:
                if rel_path not in local_manifest.files:
                    remote_path = f"{remote_dir}/{rel_path}"
                    try:
                        await self.connector.execute(f"rm {remote_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {remote_path}: {e}")
        
        return local_manifest
    
    async def sync_from_remote(
        self,
        remote_dir: str,
        local_dir: str,
        exclude_patterns: List[str] = None,
        delete_local: bool = False,
        progress_callback: Optional[Callable[[TransferProgress], None]] = None
    ) -> SyncManifest:
        """
        Synchronize remote directory to local.
        
        Args:
            remote_dir: Remote directory path
            local_dir: Local directory path
            exclude_patterns: Patterns to exclude
            delete_local: Delete local files not present remotely
            progress_callback: Progress callback
            
        Returns:
            SyncManifest of synchronized files
        """
        exclude_patterns = exclude_patterns or []
        
        # Build manifests
        remote_manifest = await self._build_remote_manifest(remote_dir)
        local_manifest = await self._build_local_manifest(local_dir, exclude_patterns)
        
        # Determine files to download
        files_to_download = []
        for rel_path, remote_info in remote_manifest.files.items():
            if rel_path not in local_manifest.files:
                files_to_download.append(rel_path)
            else:
                local_info = local_manifest.files[rel_path]
                if remote_info['size'] != local_info['size']:
                    files_to_download.append(rel_path)
        
        # Transfer files
        total_size = sum(
            remote_manifest.files[p]['size'] for p in files_to_download
        )
        transferred = 0
        
        async def progress_wrapper(bytes_done: int, total: int):
            if progress_callback:
                progress_callback(TransferProgress(
                    source=remote_dir,
                    destination="sync",
                    bytes_transferred=transferred + bytes_done,
                    total_bytes=total_size,
                    percentage=(transferred + bytes_done) / total_size * 100 if total_size > 0 else 100,
                    speed_mbps=0,
                ))
        
        for rel_path in files_to_download:
            remote_path = f"{remote_dir}/{rel_path}"
            local_path = os.path.join(local_dir, rel_path)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            await self.connector.download_file(
                remote_path, local_path, progress_wrapper
            )
            transferred += remote_manifest.files[rel_path]['size']
        
        # Handle deletions if requested
        if delete_local:
            for rel_path in local_manifest.files:
                if rel_path not in remote_manifest.files:
                    local_path = os.path.join(local_dir, rel_path)
                    try:
                        os.remove(local_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete {local_path}: {e}")
        
        return remote_manifest
    
    async def stage_job_data(
        self,
        job_id: str,
        input_files: List[str],
        output_patterns: List[str],
        local_work_dir: str,
        remote_work_dir: str
    ) -> Dict[str, str]:
        """
        Stage data for a job.
        
        Args:
            job_id: Job ID
            input_files: List of input file paths (relative to local_work_dir)
            output_patterns: Output file patterns to retrieve
            local_work_dir: Local working directory
            remote_work_dir: Remote working directory
            
        Returns:
            Dictionary with 'input_manifest' and 'output_manifest' paths
        """
        # Create remote work directory
        await self.connector.execute(f"mkdir -p {remote_work_dir}")
        
        # Upload input files
        upload_pairs = [
            (os.path.join(local_work_dir, f), f"{remote_work_dir}/{f}")
            for f in input_files
        ]
        
        upload_results = await self.upload_files(upload_pairs)
        
        if not all(upload_results.values()):
            failed = [f for f, success in upload_results.items() if not success]
            raise DataTransferError(f"Failed to upload files: {failed}")
        
        # Save manifests
        input_manifest = SyncManifest()
        for f in input_files:
            local_path = os.path.join(local_work_dir, f)
            stat = os.stat(local_path)
            input_manifest.add_file(f, stat.st_size, stat.st_mtime)
        
        manifest_path = os.path.join(local_work_dir, f"{job_id}_input_manifest.json")
        with open(manifest_path, 'w') as mf:
            import json
            json.dump(input_manifest.to_dict(), mf)
        
        return {
            'input_manifest': manifest_path,
            'remote_work_dir': remote_work_dir,
        }
    
    async def retrieve_job_output(
        self,
        job_id: str,
        output_patterns: List[str],
        local_work_dir: str,
        remote_work_dir: str,
        delete_remote: bool = True
    ) -> SyncManifest:
        """
        Retrieve job output files.
        
        Args:
            job_id: Job ID
            output_patterns: Output file patterns to retrieve
            local_work_dir: Local working directory
            remote_work_dir: Remote working directory
            delete_remote: Whether to delete remote files after retrieval
            
        Returns:
            SyncManifest of retrieved files
        """
        # Find output files
        remote_files = []
        for pattern in output_patterns:
            result = await self.connector.execute(
                f"find {remote_work_dir} -name '{pattern}' -type f"
            )
            if result['exit_code'] == 0:
                remote_files.extend(result['stdout'].strip().split('\n'))
        
        remote_files = [f for f in remote_files if f]
        
        # Download files
        download_pairs = []
        for remote_path in remote_files:
            rel_path = os.path.relpath(remote_path, remote_work_dir)
            local_path = os.path.join(local_work_dir, rel_path)
            download_pairs.append((remote_path, local_path))
        
        await self.download_files(download_pairs)
        
        # Build manifest
        manifest = SyncManifest()
        for remote_path in remote_files:
            rel_path = os.path.relpath(remote_path, remote_work_dir)
            local_path = os.path.join(local_work_dir, rel_path)
            if os.path.exists(local_path):
                stat = os.stat(local_path)
                manifest.add_file(rel_path, stat.st_size, stat.st_mtime)
        
        # Cleanup remote if requested
        if delete_remote:
            await self.connector.execute(f"rm -rf {remote_work_dir}")
        
        return manifest
    
    async def _upload_with_progress(
        self,
        local_path: str,
        remote_path: str,
        progress_callback: Optional[Callable[[TransferProgress], None]]
    ):
        """Upload with progress tracking."""
        start_time = datetime.now()
        
        def callback(bytes_done: int, total: int):
            if progress_callback:
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = (bytes_done / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                
                progress_callback(TransferProgress(
                    source=local_path,
                    destination=remote_path,
                    bytes_transferred=bytes_done,
                    total_bytes=total,
                    percentage=(bytes_done / total * 100) if total > 0 else 100,
                    speed_mbps=speed,
                    eta_seconds=(total - bytes_done) / (bytes_done / elapsed) if bytes_done > 0 and elapsed > 0 else None,
                ))
        
        await self.connector.upload_file(local_path, remote_path, callback)
    
    async def _download_with_progress(
        self,
        remote_path: str,
        local_path: str,
        progress_callback: Optional[Callable[[TransferProgress], None]]
    ):
        """Download with progress tracking."""
        start_time = datetime.now()
        
        def callback(bytes_done: int, total: int):
            if progress_callback:
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = (bytes_done / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                
                progress_callback(TransferProgress(
                    source=remote_path,
                    destination=local_path,
                    bytes_transferred=bytes_done,
                    total_bytes=total,
                    percentage=(bytes_done / total * 100) if total > 0 else 100,
                    speed_mbps=speed,
                    eta_seconds=(total - bytes_done) / (bytes_done / elapsed) if bytes_done > 0 and elapsed > 0 else None,
                ))
        
        await self.connector.download_file(remote_path, local_path, callback)
    
    async def _build_local_manifest(
        self,
        local_dir: str,
        exclude_patterns: List[str]
    ) -> SyncManifest:
        """Build manifest of local directory."""
        import fnmatch
        
        manifest = SyncManifest()
        
        for root, dirs, files in os.walk(local_dir):
            # Filter excluded directories
            dirs[:] = [
                d for d in dirs
                if not any(fnmatch.fnmatch(d, p) for p in exclude_patterns)
            ]
            
            for filename in files:
                if any(fnmatch.fnmatch(filename, p) for p in exclude_patterns):
                    continue
                
                filepath = os.path.join(root, filename)
                relpath = os.path.relpath(filepath, local_dir)
                
                stat = os.stat(filepath)
                manifest.add_file(
                    relpath.replace('\\', '/'),
                    stat.st_size,
                    stat.st_mtime
                )
        
        return manifest
    
    async def _build_remote_manifest(self, remote_dir: str) -> SyncManifest:
        """Build manifest of remote directory."""
        manifest = SyncManifest()
        
        result = await self.connector.execute(
            f"find {remote_dir} -type f -printf '%P|%s|%T@\\n' 2>/dev/null || "
            f"find {remote_dir} -type f -exec stat -c '%n|%s|%Y' {{}} \\;"
        )
        
        if result['exit_code'] != 0:
            return manifest
        
        for line in result['stdout'].strip().split('\n'):
            if '|' not in line:
                continue
            
            parts = line.split('|')
            if len(parts) >= 3:
                relpath = parts[0]
                try:
                    size = int(parts[1])
                    mtime = float(parts[2])
                    manifest.add_file(relpath, size, mtime)
                except ValueError:
                    continue
        
        return manifest
    
    async def _calculate_local_checksum(self, filepath: str) -> str:
        """Calculate MD5 checksum of local file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def _calculate_remote_checksum(self, filepath: str) -> str:
        """Calculate MD5 checksum of remote file."""
        result = await self.connector.execute(f"md5sum {filepath} 2>/dev/null || md5 -r {filepath}")
        if result['exit_code'] == 0:
            return result['stdout'].split()[0]
        return ""
