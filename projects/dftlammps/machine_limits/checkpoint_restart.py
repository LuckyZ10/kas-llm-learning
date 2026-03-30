#!/usr/bin/env python3
"""
checkpoint_restart.py - Checkpoint and restart functionality

Fault-tolerant checkpointing with incremental checkpoints,
async I/O, and automatic recovery for extreme-scale simulations.

Author: DFT-LAMMPS Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import pickle
import json
import time
import hashlib
import shutil
from pathlib import Path
import threading
import queue
from datetime import datetime
import warnings
import os
import sys

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

logger = logging.getLogger(__name__)


class CheckpointFormat(Enum):
    """Checkpoint file formats"""
    PICKLE = "pickle"
    NPZ = "npz"
    HDF5 = "hdf5"
    ZARR = "zarr"
    CUSTOM = "custom"


class CheckpointLevel(Enum):
    """Checkpoint detail levels"""
    FULL = "full"           # Complete state
    ESSENTIAL = "essential" # Essential state only
    INCREMENTAL = "incremental"  # Changes since last checkpoint
    ROLLBACK = "rollback"   # Minimal state for restart


@dataclass
class CheckpointConfig:
    """Checkpoint configuration"""
    # Frequency
    checkpoint_steps: int = 1000
    checkpoint_time_min: float = 30.0
    
    # Format
    format: CheckpointFormat = CheckpointFormat.HDF5
    compression: str = "gzip"
    compression_level: int = 4
    
    # Storage
    checkpoint_dir: str = "./checkpoints"
    max_checkpoints: int = 5
    keep_essential: bool = True
    
    # Incremental
    enable_incremental: bool = True
    full_checkpoint_every: int = 5
    
    # Async
    enable_async: bool = True
    async_queue_size: int = 2
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    failure_detection_timeout: float = 60.0
    
    # Metadata
    include_timestamp: bool = True
    include_git_version: bool = True
    include_hostname: bool = True


class IncrementalCheckpoint:
    """
    Incremental checkpoint manager
    
    Stores only changes since last checkpoint to reduce
    I/O overhead for large-scale simulations.
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_count = 0
        self.last_full_checkpoint = None
        self.last_state_hash = {}
        self.incremental_data = {}
    
    def compute_delta(self, current_state: Dict, 
                     reference_state: Dict = None) -> Dict:
        """
        Compute incremental delta from reference state
        
        Args:
            current_state: Current simulation state
            reference_state: Reference state (last checkpoint)
            
        Returns:
            Dictionary of changed items only
        """
        if reference_state is None:
            reference_state = self.last_full_checkpoint
        
        if reference_state is None:
            return current_state
        
        delta = {}
        
        for key, value in current_state.items():
            if key not in reference_state:
                delta[key] = {'type': 'new', 'value': value}
            elif isinstance(value, np.ndarray):
                # For arrays, compute which elements changed
                ref_value = reference_state[key]
                if value.shape != ref_value.shape:
                    delta[key] = {'type': 'replace', 'value': value}
                else:
                    changed = ~np.isclose(value, ref_value)
                    if np.any(changed):
                        delta[key] = {
                            'type': 'modify',
                            'indices': np.where(changed),
                            'values': value[changed]
                        }
            elif value != reference_state[key]:
                delta[key] = {'type': 'update', 'value': value}
        
        return delta
    
    def apply_delta(self, base_state: Dict, delta: Dict) -> Dict:
        """
        Apply incremental delta to base state
        
        Args:
            base_state: Base state to modify
            delta: Incremental changes
            
        Returns:
            Updated state
        """
        new_state = base_state.copy()
        
        for key, change in delta.items():
            change_type = change.get('type')
            
            if change_type in ('new', 'replace', 'update'):
                new_state[key] = change['value']
            elif change_type == 'modify':
                if key in new_state:
                    new_state[key] = new_state[key].copy()
                    indices = change['indices']
                    new_state[key][indices] = change['values']
        
        return new_state
    
    def should_create_full(self) -> bool:
        """Determine if full checkpoint should be created"""
        if self.last_full_checkpoint is None:
            return True
        
        return self.checkpoint_count % self.config.full_checkpoint_every == 0
    
    def update_reference(self, state: Dict):
        """Update reference state for future deltas"""
        self.last_full_checkpoint = {
            k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in state.items()
        }


class AsyncCheckpointWriter:
    """
    Asynchronous checkpoint writer
    
    Performs checkpoint I/O in background thread to
    minimize impact on simulation performance.
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.queue = queue.Queue(maxsize=config.async_queue_size)
        self.writer_thread = None
        self.running = False
        self.completed = 0
        self.errors = []
    
    def start(self):
        """Start background writer thread"""
        self.running = True
        self.writer_thread = threading.Thread(target=self._writer_loop)
        self.writer_thread.start()
    
    def stop(self):
        """Stop background writer"""
        self.running = False
        if self.writer_thread:
            self.queue.put(None)  # Sentinel
            self.writer_thread.join()
    
    def _writer_loop(self):
        """Background writer loop"""
        while self.running:
            task = self.queue.get()
            if task is None:
                break
            
            try:
                self._write_checkpoint(task)
                self.completed += 1
            except Exception as e:
                logger.error(f"Checkpoint write failed: {e}")
                self.errors.append((task.get('filename'), str(e)))
    
    def _write_checkpoint(self, task: Dict):
        """Write checkpoint to disk"""
        filename = task['filename']
        data = task['data']
        format = task.get('format', self.config.format)
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        if format == CheckpointFormat.PICKLE:
            with open(filename, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        elif format == CheckpointFormat.NPZ:
            np.savez_compressed(filename, **data)
        
        elif format == CheckpointFormat.HDF5 and HAS_H5PY:
            with h5py.File(filename, 'w') as f:
                self._save_to_hdf5(f, data)
        
        elif format == CheckpointFormat.ZARR and HAS_ZARR:
            store = zarr.DirectoryStore(filename)
            root = zarr.group(store=store, overwrite=True)
            self._save_to_zarr(root, data)
    
    def _save_to_hdf5(self, group, data: Dict, prefix: str = ''):
        """Recursively save to HDF5"""
        for key, value in data.items():
            full_key = f"{prefix}/{key}" if prefix else key
            
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._save_to_hdf5(subgroup, value, full_key)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value,
                                   compression=self.config.compression,
                                   compression_opts=self.config.compression_level)
            elif isinstance(value, (int, float, str, bool)):
                group.attrs[key] = value
            else:
                # Serialize other types
                group.attrs[key] = pickle.dumps(value)
    
    def _save_to_zarr(self, group, data: Dict):
        """Recursively save to Zarr"""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._save_to_zarr(subgroup, value)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value,
                                   compression='zstd',
                                   chunks=True)
            else:
                group.attrs[key] = value
    
    def submit(self, filename: str, data: Dict, 
               format: CheckpointFormat = None) -> bool:
        """
        Submit checkpoint for async writing
        
        Args:
            filename: Checkpoint filename
            data: Checkpoint data
            format: Checkpoint format
            
        Returns:
            True if submitted, False if queue full
        """
        task = {
            'filename': filename,
            'data': data,
            'format': format or self.config.format
        }
        
        try:
            self.queue.put_nowait(task)
            return True
        except queue.Full:
            return False


class FaultToleranceManager:
    """
    Fault tolerance manager for handling failures
    
    Detects failures, manages checkpoint recovery,
    and handles process restarts.
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.failed_processes = set()
        self.last_heartbeat = {}
        self.recovery_count = 0
    
    def record_heartbeat(self, rank: int, timestamp: float = None):
        """Record process heartbeat"""
        if timestamp is None:
            timestamp = time.time()
        self.last_heartbeat[rank] = timestamp
    
    def check_failures(self) -> List[int]:
        """
        Check for failed processes
        
        Returns:
            List of failed process ranks
        """
        current_time = time.time()
        timeout = self.config.failure_detection_timeout
        
        failed = []
        for rank, last_time in self.last_heartbeat.items():
            if current_time - last_time > timeout:
                if rank not in self.failed_processes:
                    failed.append(rank)
                    self.failed_processes.add(rank)
                    logger.warning(f"Detected failure of process {rank}")
        
        return failed
    
    def recover_from_failure(self, failed_ranks: List[int],
                            checkpoint_manager: 'CheckpointManager') -> Optional[Dict]:
        """
        Attempt to recover from failures
        
        Args:
            failed_ranks: List of failed process ranks
            checkpoint_manager: Checkpoint manager for recovery
            
        Returns:
            Recovered state or None
        """
        logger.info(f"Attempting recovery from failure of ranks {failed_ranks}")
        
        # Find most recent valid checkpoint
        checkpoint = checkpoint_manager.find_latest_valid()
        
        if checkpoint is None:
            logger.error("No valid checkpoint found for recovery")
            return None
        
        try:
            state = checkpoint_manager.load_checkpoint(checkpoint)
            self.recovery_count += 1
            logger.info(f"Recovery successful from checkpoint: {checkpoint}")
            return state
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return None
    
    def get_recovery_stats(self) -> Dict:
        """Get fault tolerance statistics"""
        return {
            'failed_processes': list(self.failed_processes),
            'recovery_count': self.recovery_count,
            'active_processes': len(self.last_heartbeat) - len(self.failed_processes)
        }


class CheckpointManager:
    """
    Main checkpoint manager
    
    Coordinates checkpoint creation, storage, and recovery
    with support for async I/O and fault tolerance.
    """
    
    def __init__(self, config: CheckpointConfig = None):
        self.config = config or CheckpointConfig()
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-managers
        self.incremental = IncrementalCheckpoint(self.config)
        self.async_writer = AsyncCheckpointWriter(self.config)
        self.fault_tolerance = FaultToleranceManager(self.config)
        
        # State
        self.checkpoint_history = []
        self.last_checkpoint_time = 0
        self.checkpoint_counter = 0
        
        # Start async writer if enabled
        if self.config.enable_async:
            self.async_writer.start()
    
    def create_checkpoint(self, state: Dict, step: int,
                         level: CheckpointLevel = None) -> str:
        """
        Create checkpoint
        
        Args:
            state: Simulation state to checkpoint
            step: Current simulation step
            level: Checkpoint detail level
            
        Returns:
            Checkpoint filename
        """
        if level is None:
            level = CheckpointLevel.ESSENTIAL
        
        # Determine checkpoint type
        is_full = self.incremental.should_create_full()
        
        if is_full or level == CheckpointLevel.FULL:
            checkpoint_data = self._prepare_full_checkpoint(state, step)
            self.incremental.update_reference(state)
        else:
            delta = self.incremental.compute_delta(state)
            checkpoint_data = self._prepare_incremental_checkpoint(delta, step)
        
        self.incremental.checkpoint_count += 1
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_type = "full" if is_full else "incremental"
        filename = self.checkpoint_dir / f"checkpoint_{checkpoint_type}_step{step}_{timestamp}.chk"
        
        # Write checkpoint
        if self.config.enable_async:
            success = self.async_writer.submit(str(filename), checkpoint_data)
            if not success:
                logger.warning("Async queue full, writing synchronously")
                self._write_sync(str(filename), checkpoint_data)
        else:
            self._write_sync(str(filename), checkpoint_data)
        
        # Update history
        self.checkpoint_history.append({
            'filename': str(filename),
            'step': step,
            'time': time.time(),
            'type': checkpoint_type,
            'level': level.value
        })
        
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter += 1
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Created checkpoint: {filename} ({checkpoint_type})")
        
        return str(filename)
    
    def _prepare_full_checkpoint(self, state: Dict, step: int) -> Dict:
        """Prepare full checkpoint data"""
        checkpoint = {
            'step': step,
            'type': 'full',
            'state': state,
            'metadata': self._get_metadata()
        }
        return checkpoint
    
    def _prepare_incremental_checkpoint(self, delta: Dict, step: int) -> Dict:
        """Prepare incremental checkpoint data"""
        checkpoint = {
            'step': step,
            'type': 'incremental',
            'delta': delta,
            'metadata': self._get_metadata()
        }
        return checkpoint
    
    def _get_metadata(self) -> Dict:
        """Get checkpoint metadata"""
        metadata = {
            'timestamp': time.time(),
            'python_version': sys.version,
        }
        
        if self.config.include_hostname:
            import socket
            metadata['hostname'] = socket.gethostname()
        
        if self.config.include_git_version:
            try:
                import subprocess
                git_hash = subprocess.check_output(
                    ['git', 'rev-parse', '--short', 'HEAD'],
                    cwd=str(Path(__file__).parent)
                ).decode().strip()
                metadata['git_hash'] = git_hash
            except:
                pass
        
        return metadata
    
    def _write_sync(self, filename: str, data: Dict):
        """Write checkpoint synchronously"""
        self.async_writer._write_checkpoint({
            'filename': filename,
            'data': data,
            'format': self.config.format
        })
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only most recent"""
        if len(self.checkpoint_history) <= self.config.max_checkpoints:
            return
        
        # Sort by time
        sorted_history = sorted(self.checkpoint_history, 
                               key=lambda x: x['time'],
                               reverse=True)
        
        # Keep essential checkpoints
        to_keep = set()
        
        if self.config.keep_essential:
            # Keep most recent full checkpoint
            full_checkpoints = [h for h in sorted_history if h['type'] == 'full']
            if full_checkpoints:
                to_keep.add(full_checkpoints[0]['filename'])
        
        # Keep most recent checkpoints
        for h in sorted_history[:self.config.max_checkpoints]:
            to_keep.add(h['filename'])
        
        # Remove others
        for h in sorted_history[self.config.max_checkpoints:]:
            if h['filename'] not in to_keep:
                try:
                    Path(h['filename']).unlink(missing_ok=True)
                    logger.debug(f"Removed old checkpoint: {h['filename']}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint: {e}")
        
        # Update history
        self.checkpoint_history = [h for h in self.checkpoint_history 
                                   if h['filename'] in to_keep]
    
    def load_checkpoint(self, filename: str) -> Dict:
        """
        Load checkpoint from file
        
        Args:
            filename: Checkpoint filename
            
        Returns:
            Checkpoint state
        """
        format = self._detect_format(filename)
        
        if format == CheckpointFormat.PICKLE:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        
        elif format == CheckpointFormat.NPZ:
            data = dict(np.load(filename, allow_pickle=True))
        
        elif format == CheckpointFormat.HDF5 and HAS_H5PY:
            with h5py.File(filename, 'r') as f:
                data = self._load_from_hdf5(f)
        
        elif format == CheckpointFormat.ZARR and HAS_ZARR:
            store = zarr.DirectoryStore(filename)
            root = zarr.group(store=store)
            data = self._load_from_zarr(root)
        
        else:
            raise ValueError(f"Unsupported checkpoint format: {format}")
        
        # Handle incremental checkpoints
        if data.get('type') == 'incremental':
            # Need to find base checkpoint
            base_checkpoint = self._find_base_checkpoint(filename)
            if base_checkpoint:
                base_state = self.load_checkpoint(base_checkpoint)
                state = self.incremental.apply_delta(
                    base_state.get('state', {}),
                    data.get('delta', {})
                )
                data['state'] = state
        
        return data
    
    def _detect_format(self, filename: str) -> CheckpointFormat:
        """Detect checkpoint file format"""
        path = Path(filename)
        
        if path.suffix == '.pkl' or path.suffix == '.pickle':
            return CheckpointFormat.PICKLE
        elif path.suffix == '.npz':
            return CheckpointFormat.NPZ
        elif path.suffix == '.h5' or path.suffix == '.hdf5':
            return CheckpointFormat.HDF5
        elif '.zarr' in str(path):
            return CheckpointFormat.ZARR
        else:
            return self.config.format
    
    def _load_from_hdf5(self, group) -> Dict:
        """Recursively load from HDF5"""
        data = {}
        
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                data[key] = self._load_from_hdf5(item)
            elif isinstance(item, h5py.Dataset):
                data[key] = item[()]
        
        for key, value in group.attrs.items():
            data[key] = value
        
        return data
    
    def _load_from_zarr(self, group) -> Dict:
        """Recursively load from Zarr"""
        data = {}
        
        for key in group.keys():
            item = group[key]
            if isinstance(item, zarr.Group):
                data[key] = self._load_from_zarr(item)
            elif isinstance(item, zarr.Array):
                data[key] = item[:]
        
        for key, value in group.attrs.items():
            data[key] = value
        
        return data
    
    def find_latest_valid(self) -> Optional[str]:
        """Find the most recent valid checkpoint"""
        sorted_history = sorted(self.checkpoint_history,
                               key=lambda x: x['time'],
                               reverse=True)
        
        for h in sorted_history:
            if Path(h['filename']).exists():
                return h['filename']
        
        return None
    
    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints"""
        return sorted(self.checkpoint_history, 
                     key=lambda x: x['time'],
                     reverse=True)
    
    def should_checkpoint(self, step: int, force: bool = False) -> bool:
        """
        Determine if checkpoint should be created
        
        Args:
            step: Current simulation step
            force: Force checkpoint creation
            
        Returns:
            True if checkpoint needed
        """
        if force:
            return True
        
        # Check step frequency
        if step % self.config.checkpoint_steps == 0:
            return True
        
        # Check time frequency
        elapsed_min = (time.time() - self.last_checkpoint_time) / 60
        if elapsed_min >= self.config.checkpoint_time_min:
            return True
        
        return False
    
    def close(self):
        """Cleanup checkpoint manager"""
        if self.config.enable_async:
            self.async_writer.stop()
    
    def get_stats(self) -> Dict:
        """Get checkpoint statistics"""
        return {
            'total_checkpoints': self.checkpoint_counter,
            'stored_checkpoints': len(self.checkpoint_history),
            'async_completed': self.async_writer.completed,
            'async_errors': len(self.async_writer.errors),
            'fault_tolerance': self.fault_tolerance.get_recovery_stats()
        }


def example_checkpoint_usage():
    """Example: Checkpoint and restart usage"""
    config = CheckpointConfig(
        checkpoint_dir="./test_checkpoints",
        checkpoint_steps=100,
        format=CheckpointFormat.HDF5,
        enable_async=True,
        enable_incremental=True,
        max_checkpoints=3
    )
    
    manager = CheckpointManager(config)
    
    # Simulate simulation state
    n_atoms = 100000
    state = {
        'positions': np.random.randn(n_atoms, 3),
        'velocities': np.random.randn(n_atoms, 3),
        'forces': np.random.randn(n_atoms, 3),
        'box': np.eye(3) * 100,
        'energy': -12345.67,
        'step': 1000
    }
    
    # Create checkpoints
    for step in [1000, 1100, 1200, 1300]:
        state['step'] = step
        state['positions'] += np.random.randn(n_atoms, 3) * 0.01
        
        filename = manager.create_checkpoint(state, step)
        print(f"Created checkpoint: {filename}")
    
    # List checkpoints
    print("\nAvailable checkpoints:")
    for cp in manager.list_checkpoints():
        print(f"  Step {cp['step']}: {cp['filename']} ({cp['type']})")
    
    # Load latest checkpoint
    latest = manager.find_latest_valid()
    if latest:
        loaded = manager.load_checkpoint(latest)
        print(f"\nLoaded checkpoint at step {loaded['step']}")
    
    # Print stats
    print(f"\nCheckpoint stats: {manager.get_stats()}")
    
    manager.close()
    
    return manager


if __name__ == "__main__":
    example_checkpoint_usage()
