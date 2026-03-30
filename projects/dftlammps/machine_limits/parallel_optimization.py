#!/usr/bin/env python3
"""
parallel_optimization.py - Million-core parallel optimization

Optimizations for extreme-scale parallel computing on millions of cores.
Includes communication patterns, load balancing, and topology-aware mapping.

Author: DFT-LAMMPS Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import time
from collections import defaultdict, deque
import heapq
from abc import ABC, abstractmethod
import warnings

# MPI support
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

logger = logging.getLogger(__name__)


class CommunicationPattern(Enum):
    """Communication pattern types for different network topologies"""
    ALLTOALL = auto()
    ALLTOALLV = auto()
    ALLREDUCE = auto()
    BROADCAST = auto()
    GATHER = auto()
    SCATTER = auto()
    NEAREST_NEIGHBOR = auto()
    HALO_EXCHANGE = auto()


@dataclass
class ParallelConfig:
    """Configuration for parallel optimization"""
    # Process layout
    n_procs: int = 1
    n_nodes: int = 1
    procs_per_node: int = 1
    
    # Network topology
    network_topology: str = "torus"  # torus, mesh, dragonfly, fat-tree
    torus_dimensions: Tuple[int, int, int] = (1, 1, 1)
    
    # Communication
    use_nonblocking: bool = True
    use_persistent: bool = True
    communication_threshold: int = 1024  # bytes
    
    # Load balancing
    dynamic_load_balance: bool = True
    load_balance_frequency: int = 100
    imbalance_tolerance: float = 1.2
    
    # Threading
    threads_per_proc: int = 1
    affinity_policy: str = "scatter"  # compact, scatter, balanced


class TopologyAwareMapping:
    """
    Topology-aware process mapping for minimizing communication costs
    
    Maps MPI processes to physical network topology to optimize
    communication patterns for multi-dimensional domain decomposition.
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.rank_to_coords = {}
        self.coords_to_rank = {}
        
        if HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        self._build_mapping()
    
    def _build_mapping(self):
        """Build rank to coordinate mapping based on network topology"""
        if self.config.network_topology == "torus":
            self._build_torus_mapping()
        elif self.config.network_topology == "mesh":
            self._build_mesh_mapping()
        elif self.config.network_topology == "dragonfly":
            self._build_dragonfly_mapping()
        else:
            self._build_default_mapping()
    
    def _build_torus_mapping(self):
        """Build mapping for 3D torus network"""
        nx, ny, nz = self.config.torus_dimensions
        
        for rank in range(self.size):
            # Calculate 3D coordinates from rank
            iz = rank // (nx * ny)
            iy = (rank % (nx * ny)) // nx
            ix = rank % nx
            
            self.rank_to_coords[rank] = (ix, iy, iz)
            self.coords_to_rank[(ix, iy, iz)] = rank
    
    def _build_mesh_mapping(self):
        """Build mapping for mesh network"""
        self._build_torus_mapping()  # Similar to torus but without wrap-around
    
    def _build_dragonfly_mapping(self):
        """Build mapping for Dragonfly+ network topology"""
        # Dragonfly: groups of nodes connected in all-to-all fashion
        # with intra-group and inter-group connections
        n_groups = max(1, self.size // self.config.procs_per_node)
        
        for rank in range(self.size):
            group = rank // self.config.procs_per_node
            local_rank = rank % self.config.procs_per_node
            
            self.rank_to_coords[rank] = (group, local_rank, 0)
            self.coords_to_rank[(group, local_rank, 0)] = rank
    
    def _build_default_mapping(self):
        """Default linear mapping"""
        for rank in range(self.size):
            self.rank_to_coords[rank] = (rank, 0, 0)
            self.coords_to_rank[(rank, 0, 0)] = rank
    
    def get_neighbors(self, rank: int, stencil: str = "6-point") -> List[int]:
        """
        Get neighboring ranks based on network topology
        
        Args:
            rank: Process rank
            stencil: Stencil type (6-point, 26-point, etc.)
            
        Returns:
            List of neighboring ranks
        """
        if rank not in self.rank_to_coords:
            return []
        
        coords = self.rank_to_coords[rank]
        neighbors = []
        
        if stencil == "6-point":
            # Face neighbors only
            directions = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        elif stencil == "26-point":
            # All neighbors including diagonals
            directions = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx != 0 or dy != 0 or dz != 0:
                            directions.append((dx, dy, dz))
        else:
            directions = []
        
        nx, ny, nz = self.config.torus_dimensions
        
        for d in directions:
            # Torus wrap-around
            n_coords = (
                (coords[0] + d[0]) % nx,
                (coords[1] + d[1]) % ny,
                (coords[2] + d[2]) % nz
            )
            
            if n_coords in self.coords_to_rank:
                neighbors.append(self.coords_to_rank[n_coords])
        
        return neighbors
    
    def get_distance(self, rank1: int, rank2: int) -> int:
        """
        Get Manhattan distance between two ranks in network topology
        
        Args:
            rank1, rank2: Process ranks
            
        Returns:
            Network distance (hops)
        """
        if rank1 not in self.rank_to_coords or rank2 not in self.rank_to_coords:
            return -1
        
        coords1 = self.rank_to_coords[rank1]
        coords2 = self.rank_to_coords[rank2]
        
        # Manhattan distance with torus wrap-around
        nx, ny, nz = self.config.torus_dimensions
        
        def torus_dist(a, b, n):
            d = abs(a - b)
            return min(d, n - d)
        
        dist = (
            torus_dist(coords1[0], coords2[0], nx) +
            torus_dist(coords1[1], coords2[1], ny) +
            torus_dist(coords1[2], coords2[2], nz)
        )
        
        return dist
    
    def optimize_domain_decomposition(self, n_atoms: int, box: np.ndarray) -> Tuple[int, int, int]:
        """
        Optimize domain decomposition based on network topology
        
        Args:
            n_atoms: Number of atoms
            box: Simulation box
            
        Returns:
            Optimal (nx, ny, nz) decomposition
        """
        n_procs = self.size
        
        # Find factors close to network topology
        nx, ny, nz = self.config.torus_dimensions
        
        # Optimize for aspect ratio
        box_ratios = box.diagonal() / np.min(box.diagonal())
        
        best_score = float('inf')
        best_decomp = (1, 1, n_procs)
        
        # Search for good factorization
        for px in range(1, min(n_procs + 1, 100)):
            if n_procs % px != 0:
                continue
            remaining = n_procs // px
            for py in range(1, min(remaining + 1, 100)):
                if remaining % py != 0:
                    continue
                pz = remaining // py
                
                # Score based on surface-to-volume ratio
                decomp_ratios = np.array([px, py, pz], dtype=float)
                decomp_ratios = decomp_ratios / np.min(decomp_ratios)
                
                score = np.sum((decomp_ratios - box_ratios) ** 2)
                
                # Prefer decomposition matching network topology
                if (px, py, pz) == (nx, ny, nz):
                    score *= 0.5
                
                if score < best_score:
                    best_score = score
                    best_decomp = (px, py, pz)
        
        return best_decomp


class LoadBalancer:
    """
    Dynamic load balancer for extreme-scale simulations
    
    Monitors load imbalance and performs dynamic redistribution
    of work across processes.
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.load_history = deque(maxlen=10)
        self.migration_cost = 0.0
        self.last_balance_step = 0
        
        if HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
    
    def measure_load(self, work_amount: float) -> Dict[str, float]:
        """
        Measure load across all processes
        
        Args:
            work_amount: Local work amount (e.g., number of atoms)
            
        Returns:
            Load statistics
        """
        if self.comm is None:
            return {
                'local': work_amount,
                'mean': work_amount,
                'max': work_amount,
                'min': work_amount,
                'imbalance': 1.0
            }
        
        # Gather loads from all processes
        all_loads = self.comm.gather(work_amount, root=0)
        
        stats = {}
        if self.rank == 0:
            all_loads = np.array(all_loads)
            stats = {
                'mean': np.mean(all_loads),
                'max': np.max(all_loads),
                'min': np.min(all_loads),
                'std': np.std(all_loads),
                'imbalance': np.max(all_loads) / np.mean(all_loads)
            }
        
        # Broadcast stats to all processes
        stats = self.comm.bcast(stats, root=0)
        stats['local'] = work_amount
        
        self.load_history.append(stats['imbalance'])
        
        return stats
    
    def needs_rebalancing(self, step: int, imbalance: float) -> bool:
        """
        Determine if rebalancing is needed
        
        Args:
            step: Current simulation step
            imbalance: Current imbalance factor
            
        Returns:
            True if rebalancing needed
        """
        if not self.config.dynamic_load_balance:
            return False
        
        if step - self.last_balance_step < self.config.load_balance_frequency:
            return False
        
        if imbalance < self.config.imbalance_tolerance:
            return False
        
        # Check trend in load imbalance
        if len(self.load_history) >= 3:
            recent_trend = np.mean(list(self.load_history)[-3:])
            if recent_trend < imbalance * 0.9:  # Imbalance decreasing
                return False
        
        return True
    
    def compute_migration_plan(self, loads: List[float], 
                               capacities: List[float] = None) -> Dict[int, List[Tuple[int, float]]]:
        """
        Compute atom migration plan for load balancing
        
        Args:
            loads: Current load on each process
            capacities: Maximum capacity per process (optional)
            
        Returns:
            Migration plan: {from_rank: [(to_rank, amount), ...]}
        """
        if capacities is None:
            capacities = [float('inf')] * len(loads)
        
        n_procs = len(loads)
        target_load = np.mean(loads)
        
        # Identify overloaded and underloaded processes
        overloaded = [(i, loads[i] - target_load) for i in range(n_procs) 
                      if loads[i] > target_load * 1.05]
        underloaded = [(i, target_load - loads[i]) for i in range(n_procs) 
                       if loads[i] < target_load * 0.95]
        
        # Sort by excess/deficit
        overloaded.sort(key=lambda x: -x[1])  # Most overloaded first
        underloaded.sort(key=lambda x: -x[1])  # Most capacity first
        
        # Greedy matching
        migration_plan = defaultdict(list)
        
        for src, excess in overloaded:
            for dst, capacity in underloaded:
                if excess <= 0:
                    break
                if capacity <= 0:
                    continue
                
                # Amount to migrate
                amount = min(excess, capacity)
                migration_plan[src].append((dst, amount))
                
                excess -= amount
                # Update underloaded entry
                idx = next(i for i, x in enumerate(underloaded) if x[0] == dst)
                underloaded[idx] = (dst, capacity - amount)
        
        return dict(migration_plan)
    
    def estimate_rebalancing_cost(self, migration_plan: Dict) -> float:
        """
        Estimate cost of rebalancing
        
        Args:
            migration_plan: Migration plan from compute_migration_plan
            
        Returns:
            Estimated cost in seconds
        """
        # Simplified cost model
        total_migrated = sum(
            amount for destinations in migration_plan.values()
            for _, amount in destinations
        )
        
        # Assume 1 MB/s per process pair
        bandwidth = 1e6  # bytes/s
        atom_size = 100  # bytes per atom (approximate)
        
        cost = total_migrated * atom_size / bandwidth
        
        return cost
    
    def rebalance(self, atoms_per_proc: List[int], 
                  positions: List[np.ndarray],
                  step: int) -> Tuple[List[int], List[np.ndarray]]:
        """
        Perform load rebalancing
        
        Args:
            atoms_per_proc: Number of atoms per process
            positions: Atom positions per process
            step: Current simulation step
            
        Returns:
            Updated atoms_per_proc and positions
        """
        if not HAS_MPI or self.comm is None:
            return atoms_per_proc, positions
        
        # Measure current load
        local_load = len(positions[self.rank]) if self.rank < len(positions) else 0
        stats = self.measure_load(local_load)
        
        if not self.needs_rebalancing(step, stats['imbalance']):
            return atoms_per_proc, positions
        
        logger.info(f"Load imbalance: {stats['imbalance']:.2f}, rebalancing...")
        
        # Gather all loads
        all_loads = self.comm.gather(local_load, root=0)
        
        migration_plan = {}
        if self.rank == 0:
            migration_plan = self.compute_migration_plan(all_loads)
        
        # Broadcast plan
        migration_plan = self.comm.bcast(migration_plan, root=0)
        
        # Execute migrations
        new_positions = self._execute_migrations(positions[self.rank], migration_plan)
        
        self.last_balance_step = step
        
        # Update counts
        new_counts = self.comm.allgather(len(new_positions))
        
        return new_counts, new_positions
    
    def _execute_migrations(self, local_positions: np.ndarray, 
                           migration_plan: Dict) -> np.ndarray:
        """Execute atom migrations according to plan"""
        if self.comm is None:
            return local_positions
        
        # Check if this process needs to send atoms
        if self.rank in migration_plan:
            migrations = migration_plan[self.rank]
            
            for dst, amount in migrations:
                # Select atoms to migrate (e.g., boundary atoms)
                n_migrate = int(amount)
                if n_migrate >= len(local_positions):
                    n_migrate = len(local_positions) - 1
                
                atoms_to_send = local_positions[-n_migrate:]
                local_positions = local_positions[:-n_migrate]
                
                # Send atoms
                self.comm.send(atoms_to_send, dest=dst, tag=self.rank)
        
        # Check if this process should receive atoms
        for src, destinations in migration_plan.items():
            for dst, amount in destinations:
                if dst == self.rank:
                    atoms_received = self.comm.recv(source=src, tag=src)
                    local_positions = np.vstack([local_positions, atoms_received])
        
        return local_positions


class CommunicationOptimizer:
    """
    Optimizes MPI communication patterns for million-core systems
    
    Uses non-blocking communication, persistent requests, and
    topology-aware message routing.
    """
    
    def __init__(self, config: ParallelConfig, topology: TopologyAwareMapping):
        self.config = config
        self.topology = topology
        
        if HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1
        
        # Persistent requests
        self.persistent_requests = {}
        
        # Communication statistics
        self.comm_stats = defaultdict(lambda: {'count': 0, 'bytes': 0, 'time': 0.0})
    
    def create_persistent_requests(self, neighbors: List[int], 
                                   buffer_size: int) -> Dict[int, Dict]:
        """
        Create persistent communication requests for halo exchange
        
        Args:
            neighbors: List of neighbor ranks
            buffer_size: Size of communication buffer
            
        Returns:
            Dictionary of persistent requests
        """
        if self.comm is None or not self.config.use_persistent:
            return {}
        
        requests = {}
        
        for neighbor in neighbors:
            # Send buffer
            send_buf = np.empty(buffer_size, dtype=np.float64)
            # Recv buffer
            recv_buf = np.empty(buffer_size, dtype=np.float64)
            
            # Create persistent send/recv
            send_req = self.comm.Send_init(send_buf, dest=neighbor, tag=0)
            recv_req = self.comm.Recv_init(recv_buf, source=neighbor, tag=0)
            
            requests[neighbor] = {
                'send_buf': send_buf,
                'recv_buf': recv_buf,
                'send_req': send_req,
                'recv_req': recv_req
            }
        
        self.persistent_requests = requests
        return requests
    
    def halo_exchange(self, local_data: np.ndarray, 
                     neighbors: List[int],
                     pack_func: Callable,
                     unpack_func: Callable) -> np.ndarray:
        """
        Optimized halo exchange with neighbors
        
        Args:
            local_data: Local domain data
            neighbors: List of neighbor ranks
            pack_func: Function to pack halo data
            unpack_func: Function to unpack halo data
            
        Returns:
            Data with updated halo regions
        """
        if self.comm is None:
            return local_data
        
        start_time = time.time()
        
        # Pack and send data to all neighbors
        requests = []
        recv_buffers = {}
        
        for neighbor in neighbors:
            # Pack halo data
            send_buf = pack_func(local_data, neighbor)
            
            if len(send_buf) > self.config.communication_threshold:
                # Post non-blocking receive
                recv_buf = np.empty_like(send_buf)
                recv_buffers[neighbor] = recv_buf
                
                if self.config.use_nonblocking:
                    req = self.comm.Irecv(recv_buf, source=neighbor, tag=self.rank)
                    requests.append(req)
                    
                    req = self.comm.Isend(send_buf, dest=neighbor, tag=neighbor)
                    requests.append(req)
                else:
                    self.comm.Sendrecv(
                        send_buf, dest=neighbor, sendtag=self.rank,
                        recvbuf=recv_buf, source=neighbor, recvtag=neighbor
                    )
        
        # Wait for all communications
        if self.config.use_nonblocking and requests:
            MPI.Request.Waitall(requests)
        
        # Unpack received data
        for neighbor, recv_buf in recv_buffers.items():
            local_data = unpack_func(local_data, recv_buf, neighbor)
        
        # Update statistics
        elapsed = time.time() - start_time
        self.comm_stats['halo_exchange']['count'] += 1
        self.comm_stats['halo_exchange']['time'] += elapsed
        
        return local_data
    
    def allreduce(self, local_value: np.ndarray, 
                  op: str = 'sum') -> np.ndarray:
        """
        Optimized all-reduce operation
        
        Args:
            local_value: Local contribution
            op: Reduction operation ('sum', 'max', 'min')
            
        Returns:
            Reduced value on all processes
        """
        if self.comm is None:
            return local_value
        
        start_time = time.time()
        
        # Choose MPI operation
        mpi_op = MPI.SUM
        if op == 'max':
            mpi_op = MPI.MAX
        elif op == 'min':
            mpi_op = MPI.MIN
        
        result = np.empty_like(local_value)
        self.comm.Allreduce(local_value, result, op=mpi_op)
        
        elapsed = time.time() - start_time
        self.comm_stats['allreduce']['count'] += 1
        self.comm_stats['allreduce']['bytes'] += local_value.nbytes * 2
        self.comm_stats['allreduce']['time'] += elapsed
        
        return result
    
    def get_communication_report(self) -> Dict[str, Dict]:
        """Get communication statistics report"""
        return dict(self.comm_stats)


class MillionCoreOptimizer:
    """
    Main optimizer class for million-core parallel simulations
    
    Integrates topology mapping, load balancing, and communication
    optimization for extreme-scale simulations.
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.topology = TopologyAwareMapping(config)
        self.load_balancer = LoadBalancer(config)
        self.comm_optimizer = CommunicationOptimizer(config, self.topology)
        
        # Performance tracking
        self.timers = defaultdict(float)
        self.counters = defaultdict(int)
    
    def optimize_domain_decomposition(self, n_atoms: int, 
                                      box: np.ndarray) -> Tuple[int, int, int]:
        """Get optimal domain decomposition"""
        return self.topology.optimize_domain_decomposition(n_atoms, box)
    
    def get_neighbors(self, stencil: str = "6-point") -> List[int]:
        """Get topology-aware neighbor list"""
        return self.topology.get_neighbors(
            self.topology.rank if HAS_MPI else 0, stencil
        )
    
    def exchange_halos(self, local_data: np.ndarray,
                      pack_func: Callable,
                      unpack_func: Callable) -> np.ndarray:
        """Perform optimized halo exchange"""
        neighbors = self.get_neighbors()
        return self.comm_optimizer.halo_exchange(
            local_data, neighbors, pack_func, unpack_func
        )
    
    def global_sum(self, local_value: np.ndarray) -> np.ndarray:
        """Global sum across all processes"""
        return self.comm_optimizer.allreduce(local_value, 'sum')
    
    def check_load_balance(self, local_work: float, step: int) -> bool:
        """Check if load rebalancing is needed"""
        stats = self.load_balancer.measure_load(local_work)
        return self.load_balancer.needs_rebalancing(step, stats['imbalance'])
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        report = {
            'timers': dict(self.timers),
            'counters': dict(self.counters),
            'communication': self.comm_optimizer.get_communication_report(),
            'topology': {
                'network': self.config.network_topology,
                'dimensions': self.config.torus_dimensions
            }
        }
        return report


def example_million_core_setup():
    """Example: Setup for million-core simulation"""
    config = ParallelConfig(
        n_procs=1048576,  # 1 million cores
        n_nodes=16384,    # 64 cores per node
        procs_per_node=64,
        network_topology="torus",
        torus_dimensions=(64, 64, 256),
        use_nonblocking=True,
        use_persistent=True,
        dynamic_load_balance=True
    )
    
    optimizer = MillionCoreOptimizer(config)
    
    # Example system
    n_atoms = 100000000  # 100 million atoms
    box = np.eye(3) * 1000.0  # 1000 Angstrom box
    
    # Get optimal decomposition
    decomp = optimizer.optimize_domain_decomposition(n_atoms, box)
    print(f"Optimal domain decomposition: {decomp}")
    print(f"Total processes: {np.prod(decomp)}")
    
    # Get neighbors
    neighbors = optimizer.get_neighbors("6-point")
    print(f"Number of neighbors: {len(neighbors)}")
    
    return optimizer


if __name__ == "__main__":
    example_million_core_setup()
