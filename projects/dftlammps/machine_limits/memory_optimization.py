#!/usr/bin/env python3
"""
memory_optimization.py - Memory optimization for extreme-scale simulations

Memory management strategies including out-of-core arrays, compression,
and memory pools for simulations with limited memory.

Author: DFT-LAMMPS Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging
import pickle
import gzip
import lzma
import struct
from pathlib import Path
import tempfile
import mmap
import os
import psutil
import warnings
from collections import OrderedDict
import hashlib
import time

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Compression algorithm options"""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BLOSC = "blosc"


@dataclass
class MemoryConfig:
    """Memory optimization configuration"""
    # Memory limits
    max_memory_gb: float = 64.0
    warning_threshold: float = 0.9
    critical_threshold: float = 0.95
    
    # Out-of-core settings
    enable_out_of_core: bool = True
    temp_dir: str = "/tmp"
    chunk_size: int = 10000
    
    # Compression
    compression: CompressionMethod = CompressionMethod.ZSTD
    compression_level: int = 3
    min_compress_size: int = 1024  # bytes
    
    # Caching
    cache_size_gb: float = 4.0
    cache_policy: str = "lru"  # lru, fifo, lfu
    
    # Memory pool
    enable_memory_pool: bool = True
    pool_block_size: int = 1024 * 1024  # 1 MB
    max_pool_blocks: int = 1000


class CompressionManager:
    """
    Unified compression manager supporting multiple algorithms
    
    Provides transparent compression/decompression for arrays
    with automatic algorithm selection.
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._init_compressors()
    
    def _init_compressors(self):
        """Initialize compression backends"""
        self.compressors = {}
        
        # Always available
        self.compressors[CompressionMethod.ZLIB] = self._zlib_compress
        self.compressors[CompressionMethod.GZIP] = self._gzip_compress
        
        # Optional dependencies
        try:
            import lz4.frame
            self.compressors[CompressionMethod.LZ4] = self._lz4_compress
        except ImportError:
            pass
        
        try:
            import zstandard as zstd
            self.compressors[CompressionMethod.ZSTD] = self._zstd_compress
        except ImportError:
            pass
        
        try:
            import blosc
            self.compressors[CompressionMethod.BLOSC] = self._blosc_compress
        except ImportError:
            pass
    
    def _zlib_compress(self, data: bytes) -> bytes:
        import zlib
        return zlib.compress(data, self.config.compression_level)
    
    def _zlib_decompress(self, data: bytes) -> bytes:
        import zlib
        return zlib.decompress(data)
    
    def _gzip_compress(self, data: bytes) -> bytes:
        return gzip.compress(data, self.config.compression_level)
    
    def _gzip_decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)
    
    def _lz4_compress(self, data: bytes) -> bytes:
        import lz4.frame
        return lz4.frame.compress(data)
    
    def _lz4_decompress(self, data: bytes) -> bytes:
        import lz4.frame
        return lz4.frame.decompress(data)
    
    def _zstd_compress(self, data: bytes) -> bytes:
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=self.config.compression_level)
        return cctx.compress(data)
    
    def _zstd_decompress(self, data: bytes) -> bytes:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    
    def _blosc_compress(self, data: bytes) -> bytes:
        import blosc
        return blosc.compress(data, cname='zstd', clevel=self.config.compression_level)
    
    def _blosc_decompress(self, data: bytes) -> bytes:
        import blosc
        return blosc.decompress(data)
    
    def compress(self, array: np.ndarray) -> Tuple[bytes, Tuple, str]:
        """
        Compress numpy array
        
        Returns:
            (compressed_data, original_shape, dtype)
        """
        if array.nbytes < self.config.min_compress_size:
            return array.tobytes(), array.shape, str(array.dtype)
        
        # Serialize array
        data = array.tobytes()
        shape = array.shape
        dtype = str(array.dtype)
        
        # Compress
        method = self.config.compression
        if method in self.compressors:
            compressed = self.compressors[method](data)
            return compressed, shape, dtype
        else:
            return data, shape, dtype
    
    def decompress(self, data: bytes, shape: Tuple, dtype: str) -> np.ndarray:
        """Decompress to numpy array"""
        method = self.config.compression
        
        if method in self.compressors and method != CompressionMethod.NONE:
            # Try to decompress
            try:
                decompressed = self._decompress_data(data, method)
            except:
                # If decompression fails, data might be uncompressed
                decompressed = data
        else:
            decompressed = data
        
        return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
    
    def _decompress_data(self, data: bytes, method: CompressionMethod) -> bytes:
        """Decompress raw bytes"""
        if method == CompressionMethod.ZLIB:
            return self._zlib_decompress(data)
        elif method == CompressionMethod.GZIP:
            return self._gzip_decompress(data)
        elif method == CompressionMethod.LZ4:
            return self._lz4_decompress(data)
        elif method == CompressionMethod.ZSTD:
            return self._zstd_decompress(data)
        elif method == CompressionMethod.BLOSC:
            return self._blosc_decompress(data)
        else:
            return data
    
    def get_compression_ratio(self, original: np.ndarray, compressed: bytes) -> float:
        """Calculate compression ratio"""
        return original.nbytes / len(compressed) if len(compressed) > 0 else 1.0


class OutOfCoreArray:
    """
    Out-of-core numpy array with memory-mapped file backing
    
    Allows arrays larger than physical memory by transparently
    swapping to disk.
    """
    
    def __init__(self, shape: Tuple, dtype: np.dtype = np.float64,
                 filename: str = None, config: MemoryConfig = None):
        """
        Create out-of-core array
        
        Args:
            shape: Array shape
            dtype: Data type
            filename: Optional file path (auto-generated if None)
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        self.shape = shape
        self.dtype = dtype
        self.itemsize = np.dtype(dtype).itemsize
        self.nbytes = np.prod(shape) * self.itemsize
        
        # Create temporary file
        if filename is None:
            suffix = f"_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}.tmp"
            self.filename = os.path.join(self.config.temp_dir, f"ooc_array{suffix}")
        else:
            self.filename = filename
        
        # Initialize file
        self._init_file()
        
        # Memory map
        self._mmap = None
        self._array = None
        
        # Cache for recently accessed chunks
        self.cache = OrderedDict()
        self.max_cache_items = int(self.config.cache_size_gb * 1e9 / (self.config.chunk_size * self.itemsize))
    
    def _init_file(self):
        """Initialize backing file"""
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Create file with correct size
        with open(self.filename, 'wb') as f:
            f.seek(self.nbytes - 1)
            f.write(b'\0')
    
    def _get_mmap(self) -> np.ndarray:
        """Get memory-mapped array"""
        if self._mmap is None:
            self._file = open(self.filename, 'r+b')
            self._mmap = mmap.mmap(self._file.fileno(), self.nbytes)
            self._array = np.ndarray(self.shape, dtype=self.dtype, buffer=self._mmap)
        return self._array
    
    def __getitem__(self, key):
        """Get item with caching"""
        if isinstance(key, slice) or (isinstance(key, tuple) and any(isinstance(k, slice) for k in key)):
            # Chunked access
            return self._get_mmap()[key]
        else:
            # Single element
            return self._get_mmap()[key]
    
    def __setitem__(self, key, value):
        """Set item"""
        self._get_mmap()[key] = value
    
    def __array__(self) -> np.ndarray:
        """Convert to numpy array (loads entire array to memory)"""
        return np.array(self._get_mmap())
    
    def flush(self):
        """Flush changes to disk"""
        if self._mmap is not None:
            self._mmap.flush()
    
    def close(self):
        """Close and cleanup"""
        self.flush()
        if self._mmap is not None:
            self._mmap.close()
        if hasattr(self, '_file'):
            self._file.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()
        if os.path.exists(self.filename):
            os.remove(self.filename)
    
    @classmethod
    def from_array(cls, array: np.ndarray, filename: str = None,
                   config: MemoryConfig = None) -> 'OutOfCoreArray':
        """Create OutOfCoreArray from existing numpy array"""
        ooc = cls(array.shape, array.dtype, filename, config)
        ooc[:] = array
        ooc.flush()
        return ooc


class MemoryPool:
    """
    Memory pool for efficient allocation/deallocation
    
    Reduces memory fragmentation and allocation overhead
    by reusing fixed-size blocks.
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.block_size = config.pool_block_size
        self.max_blocks = config.max_pool_blocks
        
        # Available blocks
        self.available = []
        # In-use blocks
        self.in_use = set()
        # Block storage
        self.blocks = {}
        
        self._allocate_initial_blocks()
    
    def _allocate_initial_blocks(self):
        """Pre-allocate initial pool"""
        n_initial = min(10, self.max_blocks)
        for _ in range(n_initial):
            block = np.empty(self.block_size, dtype=np.uint8)
            block_id = id(block)
            self.blocks[block_id] = block
            self.available.append(block_id)
    
    def allocate(self, size: int) -> np.ndarray:
        """
        Allocate memory from pool
        
        Args:
            size: Number of bytes needed
            
        Returns:
            Memory block as uint8 array
        """
        n_blocks = (size + self.block_size - 1) // self.block_size
        
        # Check if pool has enough blocks
        if len(self.available) < n_blocks:
            if len(self.blocks) + n_blocks <= self.max_blocks:
                # Allocate more blocks
                for _ in range(n_blocks - len(self.available)):
                    block = np.empty(self.block_size, dtype=np.uint8)
                    block_id = id(block)
                    self.blocks[block_id] = block
                    self.available.append(block_id)
            else:
                # Pool exhausted
                return np.empty(size, dtype=np.uint8)
        
        # Allocate blocks
        allocated_ids = []
        for _ in range(n_blocks):
            block_id = self.available.pop()
            allocated_ids.append(block_id)
            self.in_use.add(block_id)
        
        # Combine blocks if needed
        if n_blocks == 1:
            return self.blocks[allocated_ids[0]]
        else:
            # Return concatenated view
            blocks = [self.blocks[bid] for bid in allocated_ids]
            return np.concatenate(blocks)[:size]
    
    def free(self, array: np.ndarray):
        """
        Return memory to pool
        
        Args:
            array: Array to free
        """
        block_id = id(array)
        if block_id in self.in_use:
            self.in_use.remove(block_id)
            self.available.append(block_id)
    
    def get_stats(self) -> Dict:
        """Get pool statistics"""
        return {
            'total_blocks': len(self.blocks),
            'available': len(self.available),
            'in_use': len(self.in_use),
            'pool_memory_gb': len(self.blocks) * self.block_size / 1e9
        }


class MemoryManager:
    """
    Central memory manager for extreme-scale simulations
    
    Monitors memory usage, manages out-of-core arrays, and
    coordinates compression for memory-constrained systems.
    """
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.compression = CompressionManager(self.config)
        self.pool = MemoryPool(self.config) if self.config.enable_memory_pool else None
        
        # Memory tracking
        self.ooc_arrays = []
        self.compressed_arrays = {}
        self.peak_memory_gb = 0.0
        
        # Process memory info
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage
        
        Returns:
            Memory statistics in GB
        """
        info = self.process.memory_info()
        system = psutil.virtual_memory()
        
        usage = {
            'rss_gb': info.rss / 1e9,
            'vms_gb': info.vms / 1e9,
            'available_gb': system.available / 1e9,
            'total_gb': system.total / 1e9,
            'percent': system.percent
        }
        
        self.peak_memory_gb = max(self.peak_memory_gb, usage['rss_gb'])
        usage['peak_gb'] = self.peak_memory_gb
        
        return usage
    
    def check_memory_pressure(self) -> Tuple[bool, str]:
        """
        Check if system is under memory pressure
        
        Returns:
            (is_critical, message)
        """
        usage = self.get_memory_usage()
        
        max_mem = self.config.max_memory_gb
        rss = usage['rss_gb']
        ratio = rss / max_mem
        
        if ratio > self.config.critical_threshold:
            return True, f"CRITICAL: Memory usage {rss:.1f} GB > {max_mem * self.config.critical_threshold:.1f} GB"
        elif ratio > self.config.warning_threshold:
            return False, f"WARNING: Memory usage {rss:.1f} GB > {max_mem * self.config.warning_threshold:.1f} GB"
        else:
            return False, f"OK: Memory usage {rss:.1f} GB / {max_mem:.1f} GB"
    
    def create_array(self, shape: Tuple, dtype: np.dtype = np.float64,
                    out_of_core: bool = None) -> Union[np.ndarray, OutOfCoreArray]:
        """
        Create array with automatic out-of-core decision
        
        Args:
            shape: Array shape
            dtype: Data type
            out_of_core: Force out-of-core (auto-detect if None)
            
        Returns:
            numpy array or OutOfCoreArray
        """
        nbytes = np.prod(shape) * np.dtype(dtype).itemsize
        
        # Decide on storage type
        if out_of_core is None:
            usage = self.get_memory_usage()
            available = self.config.max_memory_gb - usage['rss_gb']
            out_of_core = (nbytes / 1e9 > available * 0.5)
        
        if out_of_core and self.config.enable_out_of_core:
            logger.info(f"Creating out-of-core array: {shape}, {nbytes/1e9:.2f} GB")
            ooc = OutOfCoreArray(shape, dtype, config=self.config)
            self.ooc_arrays.append(ooc)
            return ooc
        else:
            return np.empty(shape, dtype=dtype)
    
    def compress_array(self, array: np.ndarray, key: str = None) -> str:
        """
        Compress and store array
        
        Args:
            array: Array to compress
            key: Optional identifier (auto-generated if None)
            
        Returns:
            Key for retrieving array
        """
        if key is None:
            key = hashlib.md5(array.tobytes()).hexdigest()[:16]
        
        compressed, shape, dtype = self.compression.compress(array)
        
        self.compressed_arrays[key] = {
            'data': compressed,
            'shape': shape,
            'dtype': dtype,
            'original_size': array.nbytes,
            'compressed_size': len(compressed)
        }
        
        return key
    
    def decompress_array(self, key: str) -> np.ndarray:
        """Retrieve and decompress array"""
        if key not in self.compressed_arrays:
            raise KeyError(f"Array {key} not found")
        
        entry = self.compressed_arrays[key]
        return self.compression.decompress(
            entry['data'], entry['shape'], entry['dtype']
        )
    
    def cleanup(self):
        """Clean up all managed resources"""
        # Close out-of-core arrays
        for ooc in self.ooc_arrays:
            ooc.close()
        self.ooc_arrays.clear()
        
        # Clear compressed arrays
        self.compressed_arrays.clear()
        
        # Clear pool
        if self.pool:
            stats = self.pool.get_stats()
            logger.info(f"Pool stats: {stats}")
    
    def get_stats(self) -> Dict:
        """Get comprehensive memory statistics"""
        return {
            'memory_usage': self.get_memory_usage(),
            'ooc_arrays': len(self.ooc_arrays),
            'compressed_arrays': len(self.compressed_arrays),
            'compression_savings': sum(
                entry['original_size'] - entry['compressed_size']
                for entry in self.compressed_arrays.values()
            ) / 1e9 if self.compressed_arrays else 0.0,
            'pool_stats': self.pool.get_stats() if self.pool else None
        }


def example_memory_management():
    """Example: Memory management for large arrays"""
    config = MemoryConfig(
        max_memory_gb=16.0,
        enable_out_of_core=True,
        compression=CompressionMethod.ZSTD,
        compression_level=3
    )
    
    manager = MemoryManager(config)
    
    print("Creating large arrays...")
    
    # Create arrays larger than memory
    shape = (1000000, 1000)  # 8 GB for float64
    
    # Array 1: Out-of-core
    array1 = manager.create_array(shape, np.float64)
    array1[:] = np.random.randn(*shape)
    
    print(f"Array 1 type: {type(array1)}")
    print(f"Memory usage: {manager.get_memory_usage()}")
    
    # Compress array
    key = manager.compress_array(np.array(array1)[:1000, :1000], "sample")
    stats = manager.compression.get_compression_ratio(
        np.array(array1)[:1000, :1000],
        manager.compressed_arrays[key]['data']
    )
    print(f"Compression ratio: {stats:.2f}x")
    
    # Cleanup
    manager.cleanup()
    
    return manager


if __name__ == "__main__":
    example_memory_management()
