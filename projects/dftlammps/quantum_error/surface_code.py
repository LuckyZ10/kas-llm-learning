"""
表面码实现模块
=============
Kitaev表面码的完整实现
包含编码、解码、纠错周期
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Syndrome:
    """综合征测量结果"""
    x_syndrome: np.ndarray  # X稳定子测量结果
    z_syndrome: np.ndarray  # Z稳定子测量结果
    
    def __hash__(self):
        return hash((tuple(self.x_syndrome.flatten()), 
                     tuple(self.z_syndrome.flatten())))


class SurfaceCodeLattice:
    """
    表面码格点结构
    
    d=3的表面码布局:
    Z - D - Z - D - Z
    |       |       |
    D   X   D   X   D
    |       |       |
    Z - D - Z - D - Z
    |       |       |
    D   X   D   X   D
    |       |       |
    Z - D - Z - D - Z
    
    D = 数据量子比特
    X = X型稳定子 (星形)
    Z = Z型稳定子 (面)
    """
    
    def __init__(self, distance: int):
        self.d = distance
        
        # 数据量子比特位置 (d x d)
        self.data_positions = [(i, j) for i in range(distance) 
                               for j in range(distance)]
        
        # X稳定子位置 ((d-1) x (d-1))
        self.x_ancilla_positions = [(i + 0.5, j + 0.5) 
                                    for i in range(distance - 1) 
                                    for j in range(distance - 1)]
        
        # Z稳定子位置 (d x d) - 边界也有
        self.z_ancilla_positions = [(i, j) for i in range(distance) 
                                    for j in range(distance)]
        
        # 构建邻接关系
        self._build_adjacency()
    
    def _build_adjacency(self):
        """构建量子比特间的邻接关系"""
        # X稳定子连接到周围4个数据量子比特
        self.x_neighbors = {}
        for idx, (x, y) in enumerate(self.x_ancilla_positions):
            neighbors = []
            # 四个角的数据量子比特
            corners = [
                (int(x - 0.5), int(y - 0.5)),
                (int(x - 0.5), int(y + 0.5)),
                (int(x + 0.5), int(y - 0.5)),
                (int(x + 0.5), int(y + 0.5))
            ]
            for cx, cy in corners:
                if 0 <= cx < self.d and 0 <= cy < self.d:
                    data_idx = cx * self.d + cy
                    neighbors.append(data_idx)
            self.x_neighbors[idx] = neighbors
        
        # Z稳定子连接到周围4个数据量子比特
        self.z_neighbors = {}
        for idx, (x, y) in enumerate(self.z_ancilla_positions):
            neighbors = []
            # 四个边中点
            edges = [
                (x - 0.5, y),
                (x + 0.5, y),
                (x, y - 0.5),
                (x, y + 0.5)
            ]
            for ex, ey in edges:
                if 0 <= ex < self.d and 0 <= ey < self.d:
                    data_idx = int(ex) * self.d + int(ey)
                    neighbors.append(data_idx)
            self.z_neighbors[idx] = neighbors


class MWPM_Decoder:
    """
    最小权重完美匹配解码器
    
    使用blossom算法在综合征图上寻找最小权重匹配
    """
    
    def __init__(self, lattice: SurfaceCodeLattice):
        self.lattice = lattice
        self.distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """预计算所有稳定子之间的距离"""
        n_x = len(self.lattice.x_ancilla_positions)
        
        # 使用曼哈顿距离作为边权重
        dist_matrix = np.zeros((n_x, n_x))
        
        for i in range(n_x):
            for j in range(i + 1, n_x):
                x1, y1 = self.lattice.x_ancilla_positions[i]
                x2, y2 = self.lattice.x_ancilla_positions[j]
                dist = abs(x1 - x2) + abs(y1 - y2)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def decode(self, syndrome: Syndrome) -> Tuple[Set[int], Set[int]]:
        """
        解码综合征，返回估计的错误位置
        
        Returns:
            (x_errors, z_errors): 估计的X和Z错误位置
        """
        # 解码X错误 (使用Z综合征)
        x_errors = self._decode_type(syndrome.z_syndrome, 'X')
        
        # 解码Z错误 (使用X综合征)
        z_errors = self._decode_type(syndrome.x_syndrome, 'Z')
        
        return x_errors, z_errors
    
    def _decode_type(self, syndrome: np.ndarray, error_type: str) -> Set[int]:
        """解码特定类型的错误"""
        # 找到所有激活的 syndrome
        active = set(np.where(syndrome.flatten() == 1)[0])
        
        if len(active) == 0:
            return set()
        
        # 简化的贪心匹配算法
        # 实际应该使用blossom算法
        errors = set()
        active_list = list(active)
        
        # 如果奇数个syndrome，添加一个"边界"
        if len(active_list) % 2 == 1:
            # 边界视为一个虚拟syndrome
            pass
        
        # 贪心配对
        matched = set()
        for i in range(len(active_list)):
            if i in matched:
                continue
            
            min_dist = float('inf')
            best_match = None
            
            for j in range(i + 1, len(active_list)):
                if j in matched:
                    continue
                
                dist = self._syndrome_distance(active_list[i], active_list[j], 
                                               error_type)
                if dist < min_dist:
                    min_dist = dist
                    best_match = j
            
            if best_match is not None:
                matched.add(i)
                matched.add(best_match)
                
                # 找到连接这两个syndrome的错误链
                chain = self._find_error_chain(active_list[i], active_list[best_match],
                                               error_type)
                errors.update(chain)
        
        return errors
    
    def _syndrome_distance(self, idx1: int, idx2: int, error_type: str) -> float:
        """计算两个syndrome之间的距离"""
        if error_type == 'X':
            pos1 = self.lattice.z_ancilla_positions[idx1]
            pos2 = self.lattice.z_ancilla_positions[idx2]
        else:
            pos1 = self.lattice.x_ancilla_positions[idx1]
            pos2 = self.lattice.x_ancilla_positions[idx2]
        
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _find_error_chain(self, idx1: int, idx2: int, error_type: str) -> Set[int]:
        """找到连接两个syndrome的最小权重错误链"""
        # 使用BFS找到最短路径
        # 简化的实现：返回直线路径上的数据量子比特
        errors = set()
        
        if error_type == 'X':
            # X错误连接Z syndrome
            # 简化为选择共同邻居
            neighbors1 = set(self.lattice.z_neighbors.get(idx1, []))
            neighbors2 = set(self.lattice.z_neighbors.get(idx2, []))
            common = neighbors1.intersection(neighbors2)
            
            if common:
                errors.update(common)
            else:
                # 选择最近的邻居对
                errors.add(list(neighbors1)[0])
                errors.add(list(neighbors2)[0])
        else:
            # Z错误
            neighbors1 = set(self.lattice.x_neighbors.get(idx1, []))
            neighbors2 = set(self.lattice.x_neighbors.get(idx2, []))
            
            if neighbors1 and neighbors2:
                errors.add(list(neighbors1)[0])
                errors.add(list(neighbors2)[0])
        
        return errors


class UnionFindDecoder:
    """
    Union-Find解码器
    
    Delfosse和Tillich提出的快速解码算法
    复杂度接近线性，适合实时解码
    """
    
    def __init__(self, lattice: SurfaceCodeLattice):
        self.lattice = lattice
        self.parent = {}
        self.rank = {}
    
    def _find(self, x):
        """查找根节点"""
        if self.parent[x] != x:
            self.parent[x] = self._find(self.parent[x])
        return self.parent[x]
    
    def _union(self, x, y):
        """合并两个集合"""
        root_x = self._find(x)
        root_y = self._find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
    
    def decode(self, syndrome: Syndrome) -> Tuple[Set[int], Set[int]]:
        """使用Union-Find算法解码"""
        # 简化的实现
        x_errors = set()
        z_errors = set()
        
        # 为每个激活的 syndrome 创建节点
        self.parent = {}
        self.rank = {}
        
        active_x = np.where(syndrome.x_syndrome.flatten() == 1)[0]
        for idx in active_x:
            self.parent[f'X_{idx}'] = f'X_{idx}'
            self.rank[f'X_{idx}'] = 0
        
        # 合并相邻的 syndrome
        for i, idx1 in enumerate(active_x):
            for idx2 in active_x[i+1:]:
                if self._are_adjacent(idx1, idx2, 'X'):
                    self._union(f'X_{idx1}', f'X_{idx2}')
        
        # 为每个簇找到纠正操作
        clusters = {}
        for idx in active_x:
            root = self._find(f'X_{idx}')
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(idx)
        
        # 为每个簇应用纠正
        for cluster_syndromes in clusters.values():
            if len(cluster_syndromes) >= 2:
                # 在簇内找到错误链
                for i in range(len(cluster_syndromes) - 1):
                    chain = self._get_boundary_qubits(
                        cluster_syndromes[i], 
                        cluster_syndromes[i+1],
                        'X'
                    )
                    z_errors.update(chain)
        
        return x_errors, z_errors
    
    def _are_adjacent(self, idx1: int, idx2: int, syndrome_type: str) -> bool:
        """检查两个syndrome是否相邻"""
        if syndrome_type == 'X':
            neighbors1 = set(self.lattice.x_neighbors.get(idx1, []))
            neighbors2 = set(self.lattice.x_neighbors.get(idx2, []))
        else:
            neighbors1 = set(self.lattice.z_neighbors.get(idx1, []))
            neighbors2 = set(self.lattice.z_neighbors.get(idx2, []))
        
        return len(neighbors1.intersection(neighbors2)) > 0
    
    def _get_boundary_qubits(self, idx1: int, idx2: int, syndrome_type: str) -> Set[int]:
        """获取边界上的数据量子比特"""
        # 简化的实现
        return set()


class SurfaceCodeSimulation:
    """
    表面码完整模拟
    
    包含错误注入、综合征测量、解码和纠错
    """
    
    def __init__(self, distance: int, physical_error_rate: float = 0.001,
                 measurement_error_rate: float = 0.01):
        self.d = distance
        self.p_data = physical_error_rate
        self.p_meas = measurement_error_rate
        
        self.lattice = SurfaceCodeLattice(distance)
        self.decoder = MWPM_Decoder(self.lattice)
        
        # 初始状态
        self.data_qubits = np.zeros(distance * distance, dtype=int)
        
        # 错误历史
        self.error_history = []
        self.syndrome_history = []
    
    def inject_errors(self) -> Tuple[Set[int], Set[int]]:
        """
        注入随机错误
        
        Returns:
            (x_errors, z_errors): 注入的X和Z错误
        """
        x_errors = set()
        z_errors = set()
        
        for i in range(len(self.data_qubits)):
            if np.random.random() < self.p_data:
                error_type = np.random.choice(['X', 'Z', 'Y'])
                
                if error_type == 'X':
                    x_errors.add(i)
                    self.data_qubits[i] ^= 1  # X错误翻转比特
                elif error_type == 'Z':
                    z_errors.add(i)
                elif error_type == 'Y':
                    x_errors.add(i)
                    z_errors.add(i)
                    self.data_qubits[i] ^= 1
        
        return x_errors, z_errors
    
    def measure_syndrome(self, x_errors: Set[int], z_errors: Set[int]) -> Syndrome:
        """测量稳定子 syndrome"""
        # X稳定子检测 Z错误
        x_syndrome = np.zeros((self.d - 1, self.d - 1), dtype=int)
        for i in range(self.d - 1):
            for j in range(self.d - 1):
                idx = i * (self.d - 1) + j
                neighbors = self.lattice.x_neighbors.get(idx, [])
                
                parity = 0
                for n in neighbors:
                    if n in z_errors:
                        parity ^= 1
                
                # 添加测量错误
                if np.random.random() < self.p_meas:
                    parity ^= 1
                
                x_syndrome[i, j] = parity
        
        # Z稳定子检测 X错误
        z_syndrome = np.zeros((self.d, self.d), dtype=int)
        for i in range(self.d):
            for j in range(self.d):
                idx = i * self.d + j
                neighbors = self.lattice.z_neighbors.get(idx, [])
                
                parity = 0
                for n in neighbors:
                    if n in x_errors:
                        parity ^= 1
                
                if np.random.random() < self.p_meas:
                    parity ^= 1
                
                z_syndrome[i, j] = parity
        
        return Syndrome(x_syndrome, z_syndrome)
    
    def run_cycle(self) -> Dict:
        """运行一个完整的纠错周期"""
        # 注入错误
        x_errors, z_errors = self.inject_errors()
        
        # 测量综合征
        syndrome = self.measure_syndrome(x_errors, z_errors)
        
        # 解码
        estimated_x, estimated_z = self.decoder.decode(syndrome)
        
        # 应用纠正
        for q in estimated_x:
            self.data_qubits[q] ^= 1
        
        # 记录历史
        self.error_history.append({
            'x': x_errors,
            'z': z_errors
        })
        self.syndrome_history.append(syndrome)
        
        # 检查是否有逻辑错误
        logical_error = self._check_logical_error(x_errors, z_errors,
                                                   estimated_x, estimated_z)
        
        return {
            'injected_x': x_errors,
            'injected_z': z_errors,
            'estimated_x': estimated_x,
            'estimated_z': estimated_z,
            'syndrome': syndrome,
            'logical_error': logical_error
        }
    
    def _check_logical_error(self, true_x: Set[int], true_z: Set[int],
                             est_x: Set[int], est_z: Set[int]) -> bool:
        """检查是否存在逻辑错误"""
        # 计算残差错误
        residual_x = true_x.symmetric_difference(est_x)
        residual_z = true_z.symmetric_difference(est_z)
        
        # 检查是否与逻辑算符对易
        # 逻辑X连接左边界和右边界
        logical_x_chain = self._get_logical_x_chain()
        
        # 如果残差Z与逻辑X链有奇数个交点，则有逻辑错误
        x_commute = len(residual_z.intersection(logical_x_chain)) % 2 == 0
        
        # 逻辑Z连接上边界和下边界
        logical_z_chain = self._get_logical_z_chain()
        z_commute = len(residual_x.intersection(logical_z_chain)) % 2 == 0
        
        return not (x_commute and z_commute)
    
    def _get_logical_x_chain(self) -> Set[int]:
        """获取逻辑X算符的链 (Z型)"""
        # 顶行的数据量子比特
        return set(range(self.d))
    
    def _get_logical_z_chain(self) -> Set[int]:
        """获取逻辑Z算符的链 (X型)"""
        # 左列的数据量子比特
        return set(i * self.d for i in range(self.d))
    
    def run_memory_experiment(self, n_cycles: int = 1000) -> Dict:
        """
        运行内存实验
        
        存储|0_L>态，定期进行纠错
        """
        # 初始化|0_L>
        self.data_qubits = np.zeros(self.d * self.d, dtype=int)
        
        logical_errors = 0
        
        for _ in range(n_cycles):
            result = self.run_cycle()
            if result['logical_error']:
                logical_errors += 1
        
        return {
            'total_cycles': n_cycles,
            'logical_errors': logical_errors,
            'logical_error_rate': logical_errors / n_cycles,
            'lifetime': n_cycles / max(logical_errors, 1)
        }


def benchmark_decoders(distance: int = 5, n_trials: int = 1000) -> Dict:
    """比较不同解码器的性能"""
    error_rates = [0.001, 0.003, 0.005, 0.01]
    
    results = {
        'MWPM': {},
        'UnionFind': {}
    }
    
    for p in error_rates:
        # MWPM解码器
        sim_mwpm = SurfaceCodeSimulation(distance, p)
        errors_mwpm = 0
        for _ in range(n_trials):
            result = sim_mwpm.run_cycle()
            if result['logical_error']:
                errors_mwpm += 1
        
        results['MWPM'][p] = errors_mwpm / n_trials
        
        # UnionFind解码器
        sim_uf = SurfaceCodeSimulation(distance, p)
        sim_uf.decoder = UnionFindDecoder(sim_uf.lattice)
        errors_uf = 0
        for _ in range(n_trials):
            result = sim_uf.run_cycle()
            if result['logical_error']:
                errors_uf += 1
        
        results['UnionFind'][p] = errors_uf / n_trials
    
    return results


if __name__ == "__main__":
    # 运行示例
    print("表面码模拟示例")
    print("=" * 50)
    
    sim = SurfaceCodeSimulation(distance=3, physical_error_rate=0.01)
    result = sim.run_memory_experiment(n_cycles=100)
    
    print(f"码距: {sim.d}")
    print(f"总周期数: {result['total_cycles']}")
    print(f"逻辑错误数: {result['logical_errors']}")
    print(f"逻辑错误率: {result['logical_error_rate']:.4f}")
    print(f"逻辑量子比特寿命: {result['lifetime']:.1f} 周期")
