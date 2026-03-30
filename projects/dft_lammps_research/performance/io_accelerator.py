#!/usr/bin/env python3
"""
io_accelerator.py
=================
I/O加速模块

提供高效的文件I/O功能：
- 内存映射文件读取
- 异步I/O支持
- 批量读写优化
- 压缩缓存（lz4/zstd）

作者: Performance Optimization Expert
日期: 2026-03-22
"""

import os
import mmap
import asyncio
from typing import Dict, List, Optional, Callable, Any, Union, BinaryIO, TextIO, Iterator
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import time
import pickle
import json

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入aiofiles
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False
    logger.debug("aiofiles not available, async file operations disabled")

# 尝试导入压缩库
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


@dataclass
class IOStats:
    """I/O统计信息"""
    bytes_read: int = 0
    bytes_written: int = 0
    read_ops: int = 0
    write_ops: int = 0
    total_read_time: float = 0.0
    total_write_time: float = 0.0
    
    @property
    def read_throughput_mbps(self) -> float:
        """读取吞吐量 (MB/s)"""
        if self.total_read_time > 0:
            return (self.bytes_read / (1024 * 1024)) / self.total_read_time
        return 0.0
    
    @property
    def write_throughput_mbps(self) -> float:
        """写入吞吐量 (MB/s)"""
        if self.total_write_time > 0:
            return (self.bytes_written / (1024 * 1024)) / self.total_write_time
        return 0.0


class MemoryMappedFile:
    """
    内存映射文件
    
    使用mmap高效读取大文件。
    
    示例:
        with MemoryMappedFile("large.bin") as mmf:
            data = mmf.read_chunk(0, 1024)  # 读取前1KB
            # 或直接访问缓冲区
            buffer = mmf.buffer
    """
    
    def __init__(self, filepath: Union[str, Path], mode: str = 'r'):
        """
        初始化内存映射文件
        
        Args:
            filepath: 文件路径
            mode: 打开模式 ('r'=只读, 'r+'=读写)
        """
        self.filepath = Path(filepath)
        self.mode = mode
        self._file: Optional[BinaryIO] = None
        self._mmap: Optional[mmap.mmap] = None
        self._size: int = 0
    
    def open(self) -> 'MemoryMappedFile':
        """打开文件"""
        access = mmap.ACCESS_READ if self.mode == 'r' else mmap.ACCESS_WRITE
        
        file_mode = 'rb' if self.mode == 'r' else 'r+b'
        self._file = open(self.filepath, file_mode)
        self._size = os.fstat(self._file.fileno()).st_size
        
        if self._size > 0:
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=access)
        
        return self
    
    def close(self) -> None:
        """关闭文件"""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None
    
    def read_chunk(self, offset: int, size: int) -> bytes:
        """
        读取数据块
        
        Args:
            offset: 起始偏移
            size: 读取大小
        
        Returns:
            数据字节
        """
        if self._mmap is None:
            return b''
        return self._mmap[offset:offset + size]
    
    def read_all(self) -> bytes:
        """读取全部内容"""
        if self._mmap is None:
            return b''
        return self._mmap[:]
    
    def read_line(self, offset: int = 0) -> Optional[bytes]:
        """
        从指定位置读取一行
        
        Args:
            offset: 起始偏移
        
        Returns:
            行数据或None
        """
        if self._mmap is None:
            return None
        
        end = self._mmap.find(b'\n', offset)
        if end == -1:
            end = self._size
        else:
            end += 1  # 包含换行符
        
        return self._mmap[offset:end]
    
    def find(self, pattern: bytes, start: int = 0) -> int:
        """
        搜索模式
        
        Args:
            pattern: 搜索字节模式
            start: 起始位置
        
        Returns:
            找到的位置或-1
        """
        if self._mmap is None:
            return -1
        return self._mmap.find(pattern, start)
    
    @property
    def buffer(self) -> Optional[mmap.mmap]:
        """获取mmap缓冲区"""
        return self._mmap
    
    @property
    def size(self) -> int:
        """文件大小"""
        return self._size
    
    def __enter__(self) -> 'MemoryMappedFile':
        return self.open()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncReader:
    """
    异步文件读取器
    
    使用aiofiles实现异步I/O。
    
    示例:
        reader = AsyncReader()
        
        async def main():
            data = await reader.read_file("data.txt")
            files = ["file1.txt", "file2.txt"]
            results = await reader.read_multiple(files)
        
        asyncio.run(main())
    """
    
    def __init__(self, max_workers: int = 10):
        """
        初始化异步读取器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._stats = IOStats()
    
    async def read_file(self, filepath: Union[str, Path], 
                        mode: str = 'r',
                        chunk_size: Optional[int] = None) -> Union[str, bytes]:
        """
        异步读取文件
        
        Args:
            filepath: 文件路径
            mode: 读取模式 ('r'/'rb')
            chunk_size: 分块大小（大文件）
        
        Returns:
            文件内容
        """
        filepath = Path(filepath)
        start_time = time.time()
        
        if not AIOFILES_AVAILABLE:
            # 使用线程池作为回退
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                self._executor,
                lambda: filepath.read_bytes() if 'b' in mode else filepath.read_text()
            )
        else:
            if chunk_size and filepath.stat().st_size > chunk_size:
                # 大文件分块读取
                content = await self._read_large_file(filepath, mode, chunk_size)
            else:
                async with aiofiles.open(filepath, mode) as f:
                    content = await f.read()
        
        # 更新统计
        elapsed = time.time() - start_time
        self._stats.bytes_read += len(content) if isinstance(content, bytes) else len(content.encode())
        self._stats.read_ops += 1
        self._stats.total_read_time += elapsed
        
        return content
    
    async def _read_large_file(self, filepath: Path, mode: str, chunk_size: int) -> Union[str, bytes]:
        """分块读取大文件"""
        if not AIOFILES_AVAILABLE:
            # 使用线程池作为回退
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                lambda: filepath.read_bytes() if 'b' in mode else filepath.read_text()
            )
        
        chunks = []
        
        async with aiofiles.open(filepath, mode) as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
        
        if mode == 'rb':
            return b''.join(chunks)
        else:
            return ''.join(chunks)
    
    async def read_multiple(self, 
                           filepaths: List[Union[str, Path]],
                           mode: str = 'r') -> List[Union[str, bytes]]:
        """
        并发读取多个文件
        
        Args:
            filepaths: 文件路径列表
            mode: 读取模式
        
        Returns:
            内容列表
        """
        tasks = [self.read_file(fp, mode) for fp in filepaths]
        return await asyncio.gather(*tasks)
    
    async def read_json(self, filepath: Union[str, Path]) -> Any:
        """异步读取JSON"""
        content = await self.read_file(filepath, 'r')
        return json.loads(content)
    
    async def read_numpy(self, filepath: Union[str, Path]) -> np.ndarray:
        """异步读取NumPy数组"""
        # NumPy读取在线程池中执行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            np.load,
            filepath
        )
    
    async def write_file(self, filepath: Union[str, Path],
                        content: Union[str, bytes],
                        mode: str = 'w') -> None:
        """
        异步写入文件
        
        Args:
            filepath: 文件路径
            content: 内容
            mode: 写入模式
        """
        start_time = time.time()
        
        if not AIOFILES_AVAILABLE:
            # 使用线程池作为回退
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                lambda: Path(filepath).write_bytes(content) if 'b' in mode else Path(filepath).write_text(content)
            )
        else:
            async with aiofiles.open(filepath, mode) as f:
                await f.write(content)
        
        elapsed = time.time() - start_time
        self._stats.bytes_written += len(content) if isinstance(content, bytes) else len(content.encode())
        self._stats.write_ops += 1
        self._stats.total_write_time += elapsed
    
    def get_stats(self) -> IOStats:
        """获取I/O统计"""
        return self._stats


class BatchIO:
    """
    批量I/O优化器
    
    合并小I/O操作，提高吞吐量。
    
    示例:
        with BatchIO(batch_size=1000) as bio:
            for i in range(10000):
                bio.queue_write(f"file_{i}.txt", data[i])
            # 退出时自动批量写入
    """
    
    def __init__(self, batch_size: int = 100, max_delay_ms: float = 100):
        """
        初始化批量I/O
        
        Args:
            batch_size: 批大小
            max_delay_ms: 最大延迟（毫秒）
        """
        self.batch_size = batch_size
        self.max_delay = max_delay_ms / 1000  # 转换为秒
        
        self._write_queue: List[tuple] = []
        self._read_queue: List[tuple] = []
        self._stats = IOStats()
        self._lock = threading.Lock()
    
    def queue_write(self, filepath: Union[str, Path], 
                    content: Union[str, bytes],
                    mode: str = 'w') -> None:
        """
        排队写入
        
        Args:
            filepath: 文件路径
            content: 内容
            mode: 写入模式
        """
        with self._lock:
            self._write_queue.append((filepath, content, mode))
            
            if len(self._write_queue) >= self.batch_size:
                self._flush_writes()
    
    def queue_read(self, filepath: Union[str, Path], 
                   callback: Callable,
                   mode: str = 'r') -> None:
        """
        排队读取
        
        Args:
            filepath: 文件路径
            callback: 读取完成回调
            mode: 读取模式
        """
        with self._lock:
            self._read_queue.append((filepath, callback, mode))
            
            if len(self._read_queue) >= self.batch_size:
                self._flush_reads()
    
    def _flush_writes(self) -> None:
        """刷新写入队列"""
        with self._lock:
            if not self._write_queue:
                return
            
            batch = self._write_queue[:]
            self._write_queue.clear()
        
        start_time = time.time()
        
        # 批量写入
        for filepath, content, mode in batch:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, mode) as f:
                f.write(content)
        
        elapsed = time.time() - start_time
        total_bytes = sum(len(c) if isinstance(c, bytes) else len(c.encode()) for _, c, _ in batch)
        
        self._stats.bytes_written += total_bytes
        self._stats.write_ops += len(batch)
        self._stats.total_write_time += elapsed
        
        logger.debug(f"Batch write: {len(batch)} files, {total_bytes} bytes")
    
    def _flush_reads(self) -> None:
        """刷新读取队列"""
        with self._lock:
            if not self._read_queue:
                return
            
            batch = self._read_queue[:]
            self._read_queue.clear()
        
        start_time = time.time()
        
        # 批量读取
        for filepath, callback, mode in batch:
            with open(filepath, mode) as f:
                content = f.read()
            callback(content)
        
        elapsed = time.time() - start_time
        
        self._stats.read_ops += len(batch)
        self._stats.total_read_time += elapsed
        
        logger.debug(f"Batch read: {len(batch)} files")
    
    def flush(self) -> None:
        """强制刷新所有队列"""
        self._flush_writes()
        self._flush_reads()
    
    def get_stats(self) -> IOStats:
        """获取统计"""
        return self._stats
    
    def __enter__(self) -> 'BatchIO':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


class CompressionCache:
    """
    压缩缓存
    
    使用lz4或zstd压缩数据，减少I/O开销。
    
    示例:
        cache = CompressionCache(algorithm='zstd', level=3)
        
        # 压缩保存
        cache.save_compressed(data, "data.zst")
        
        # 解压加载
        data = cache.load_compressed("data.zst")
    """
    
    def __init__(self, algorithm: str = 'lz4', level: int = 1):
        """
        初始化压缩缓存
        
        Args:
            algorithm: 压缩算法 ('lz4', 'zstd')
            level: 压缩级别
        """
        self.algorithm = algorithm.lower()
        self.level = level
        
        if self.algorithm == 'lz4' and not LZ4_AVAILABLE:
            raise ImportError("lz4 not available, install with: pip install lz4")
        if self.algorithm == 'zstd' and not ZSTD_AVAILABLE:
            raise ImportError("zstandard not available, install with: pip install zstandard")
    
    def compress(self, data: bytes) -> bytes:
        """
        压缩数据
        
        Args:
            data: 原始数据
        
        Returns:
            压缩数据
        """
        if self.algorithm == 'lz4':
            return lz4.frame.compress(data, compression_level=self.level)
        elif self.algorithm == 'zstd':
            cctx = zstd.ZstdCompressor(level=self.level)
            return cctx.compress(data)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def decompress(self, data: bytes) -> bytes:
        """
        解压数据
        
        Args:
            data: 压缩数据
        
        Returns:
            原始数据
        """
        if self.algorithm == 'lz4':
            return lz4.frame.decompress(data)
        elif self.algorithm == 'zstd':
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def save_compressed(self, data: Any, filepath: Union[str, Path]) -> None:
        """
        压缩保存对象
        
        Args:
            data: 要保存的数据
            filepath: 文件路径
        """
        filepath = Path(filepath)
        
        # 序列化
        serialized = pickle.dumps(data)
        
        # 压缩
        compressed = self.compress(serialized)
        
        # 写入
        filepath.write_bytes(compressed)
        
        ratio = len(compressed) / len(serialized)
        logger.debug(f"Saved compressed: {filepath}, ratio: {ratio:.2%}")
    
    def load_compressed(self, filepath: Union[str, Path]) -> Any:
        """
        加载解压对象
        
        Args:
            filepath: 文件路径
        
        Returns:
            原始数据
        """
        # 读取
        compressed = Path(filepath).read_bytes()
        
        # 解压
        serialized = self.decompress(compressed)
        
        # 反序列化
        return pickle.loads(serialized)
    
    def save_array(self, arr: np.ndarray, filepath: Union[str, Path]) -> None:
        """
        压缩保存NumPy数组
        
        Args:
            arr: NumPy数组
            filepath: 文件路径
        """
        # 使用NumPy的压缩保存
        np.savez_compressed(filepath, data=arr)
    
    def load_array(self, filepath: Union[str, Path]) -> np.ndarray:
        """加载NumPy数组"""
        data = np.load(filepath)
        return data['data'] if 'data' in data else data


class IOAccelerator:
    """
    I/O加速器
    
    统一的I/O优化接口，自动选择最佳策略。
    
    示例:
        io = IOAccelerator()
        
        # 快速读取大文件
        data = io.fast_read("large_file.bin")
        
        # 异步并发读取
        results = io.concurrent_read(["file1.txt", "file2.txt"])
    """
    
    _instance: Optional['IOAccelerator'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._async_reader: Optional[AsyncReader] = None
        self._compression: Optional[CompressionCache] = None
        self._stats = IOStats()
    
    def fast_read(self, filepath: Union[str, Path], 
                  use_mmap: bool = True) -> bytes:
        """
        快速读取文件
        
        Args:
            filepath: 文件路径
            use_mmap: 是否使用内存映射
        
        Returns:
            文件内容
        """
        filepath = Path(filepath)
        file_size = filepath.stat().st_size
        
        start_time = time.time()
        
        if use_mmap and file_size > 1024 * 1024:  # > 1MB
            # 大文件使用内存映射
            with MemoryMappedFile(filepath) as mmf:
                content = mmf.read_all()
        else:
            # 小文件直接读取
            content = filepath.read_bytes()
        
        elapsed = time.time() - start_time
        self._stats.bytes_read += len(content)
        self._stats.read_ops += 1
        self._stats.total_read_time += elapsed
        
        return content
    
    def fast_write(self, filepath: Union[str, Path], 
                   content: Union[str, bytes],
                   use_buffer: bool = True) -> None:
        """
        快速写入文件
        
        Args:
            filepath: 文件路径
            content: 内容
            use_buffer: 是否使用缓冲
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        mode = 'wb' if isinstance(content, bytes) else 'w'
        
        if use_buffer and len(content) > 1024 * 1024:
            # 大文件使用缓冲
            with open(filepath, mode, buffering=1024*1024) as f:
                f.write(content)
        else:
            filepath.write_bytes(content) if isinstance(content, bytes) else filepath.write_text(content)
        
        elapsed = time.time() - start_time
        self._stats.bytes_written += len(content) if isinstance(content, bytes) else len(content.encode())
        self._stats.write_ops += 1
        self._stats.total_write_time += elapsed
    
    def concurrent_read(self, 
                       filepaths: List[Union[str, Path]],
                       max_workers: int = 10) -> List[bytes]:
        """
        并发读取多个文件
        
        Args:
            filepaths: 文件路径列表
            max_workers: 最大线程数
        
        Returns:
            内容列表
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.fast_read, filepaths))
        return results
    
    def stream_lines(self, filepath: Union[str, Path],
                     chunk_size: int = 8192) -> Iterator[str]:
        """
        流式读取行
        
        Args:
            filepath: 文件路径
            chunk_size: 块大小
        
        Yields:
            行字符串
        """
        buffer = ''
        
        with open(filepath, 'r', buffering=chunk_size) as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                buffer += chunk
                lines = buffer.split('\n')
                buffer = lines.pop()  # 保留不完整的行
                
                for line in lines:
                    yield line
        
        if buffer:
            yield buffer
    
    def get_stats(self) -> IOStats:
        """获取I/O统计"""
        return self._stats


# 便捷函数

def fast_read(filepath: Union[str, Path], use_mmap: bool = True) -> bytes:
    """快速读取文件"""
    io = IOAccelerator()
    return io.fast_read(filepath, use_mmap)


def fast_write(filepath: Union[str, Path], content: Union[str, bytes]) -> None:
    """快速写入文件"""
    io = IOAccelerator()
    return io.fast_write(filepath, content)


async def async_read(filepath: Union[str, Path]) -> str:
    """异步读取文件"""
    reader = AsyncReader()
    return await reader.read_file(filepath)
