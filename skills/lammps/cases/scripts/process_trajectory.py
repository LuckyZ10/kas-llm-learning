#!/usr/bin/env python3
"""
LAMMPS数据处理工具
process_trajectory.py - 轨迹后处理
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import argparse
from collections import defaultdict

class TrajectoryProcessor:
    """LAMMPS轨迹处理器"""
    
    def __init__(self, dump_file):
        self.dump_file = dump_file
        self.frames = []
        self.parse_dump()
    
    def parse_dump(self):
        """解析dump文件"""
        with open(self.dump_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            if lines[i].startswith('ITEM: TIMESTEP'):
                frame = {'timestep': int(lines[i+1].strip())}
                i += 2
                
                # 原子数
                if lines[i].startswith('ITEM: NUMBER OF ATOMS'):
                    frame['natoms'] = int(lines[i+1].strip())
                    i += 2
                
                # Box边界
                if lines[i].startswith('ITEM: BOX BOUNDS'):
                    box = []
                    for j in range(3):
                        parts = lines[i+1+j].split()
                        box.append([float(parts[0]), float(parts[1])])
                    frame['box'] = np.array(box)
                    i += 4
                
                # 原子数据
                if lines[i].startswith('ITEM: ATOMS'):
                    headers = lines[i].replace('ITEM: ATOMS ', '').split()
                    atoms = []
                    for j in range(frame['natoms']):
                        data = lines[i+1+j].split()
                        atom = dict(zip(headers, [float(x) for x in data]))
                        atoms.append(atom)
                    frame['atoms'] = atoms
                    i += frame['natoms'] + 1
                
                self.frames.append(frame)
            else:
                i += 1
        
        print(f"Loaded {len(self.frames)} frames")
    
    def compute_rdf(self, frame_idx=-1, nbins=100, rmax=None):
        """计算RDF"""
        frame = self.frames[frame_idx]
        positions = np.array([[a['x'], a['y'], a['z']] for a in frame['atoms']])
        
        if rmax is None:
            rmax = min(frame['box'][:, 1] - frame['box'][:, 0]) / 2
        
        # 计算所有距离
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)  # 排除自身
        
        # 创建直方图
        hist, bin_edges = np.histogram(distances[distances < rmax], bins=nbins, range=(0, rmax))
        
        # 归一化
        dr = bin_edges[1] - bin_edges[0]
        r = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        rho = len(positions) / np.prod(frame['box'][:, 1] - frame['box'][:, 0])
        shell_volume = 4 * np.pi * r**2 * dr
        
        g_r = hist / (shell_volume * rho * len(positions))
        
        return r, g_r
    
    def compute_msd(self, atom_type=None):
        """计算MSD"""
        if len(self.frames) < 2:
            return None, None
        
        ref_positions = {}
        for atom in self.frames[0]['atoms']:
            if atom_type is None or atom.get('type') == atom_type:
                ref_positions[int(atom['id'])] = np.array([atom['x'], atom['y'], atom['z']])
        
        msd_values = []
        timesteps = []
        
        for frame in self.frames:
            displacements = []
            for atom in frame['atoms']:
                atom_id = int(atom['id'])
                if atom_id in ref_positions:
                    pos = np.array([atom['x'], atom['y'], atom['z']])
                    dr = pos - ref_positions[atom_id]
                    displacements.append(np.sum(dr**2))
            
            if displacements:
                msd_values.append(np.mean(displacements))
                timesteps.append(frame['timestep'])
        
        return np.array(timesteps), np.array(msd_values)
    
    def compute_cluster_analysis(self, frame_idx=-1, cutoff=3.5):
        """团簇分析"""
        frame = self.frames[frame_idx]
        positions = np.array([[a['x'], a['y'], a['z']] for a in frame['atoms']])
        
        # 距离矩阵
        distances = cdist(positions, positions)
        
        # 构建邻接表
        neighbors = distances < cutoff
        np.fill_diagonal(neighbors, False)
        
        # BFS找连通分量
        visited = np.zeros(len(positions), dtype=bool)
        clusters = []
        
        for i in range(len(positions)):
            if not visited[i]:
                cluster = []
                queue = [i]
                visited[i] = True
                
                while queue:
                    current = queue.pop(0)
                    cluster.append(current)
                    
                    for j in np.where(neighbors[current])[0]:
                        if not visited[j]:
                            visited[j] = True
                            queue.append(j)
                
                clusters.append(cluster)
        
        return clusters
    
    def compute_voronoi_analysis(self, frame_idx=-1):
        """Voronoi分析"""
        frame = self.frames[frame_idx]
        positions = np.array([[a['x'], a['y'], a['z']] for a in frame['atoms']])
        
        # 2D Voronoi (xy平面)
        vor = Voronoi(positions[:, :2])
        
        # 计算每个 Voronoi 单元的体积(面积)
        volumes = []
        for region in vor.regions:
            if -1 not in region and len(region) > 0:
                # 计算多边形面积
                vertices = vor.vertices[region]
                area = 0.5 * abs(np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) - 
                                np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
                volumes.append(area)
        
        return vor, volumes
    
    def identify_local_structure(self, frame_idx=-1, cutoff=3.5):
        """识别局部结构 (CNA)"""
        frame = self.frames[frame_idx]
        positions = np.array([[a['x'], a['y'], a['z']] for a in frame['atoms']])
        
        structures = []
        
        for i, pos in enumerate(positions):
            # 找到邻居
            distances = np.linalg.norm(positions - pos, axis=1)
            neighbors = np.where((distances < cutoff) & (distances > 0.1))[0]
            
            if len(neighbors) == 12:
                # 简化的CNA - 检查共近邻
                # 这里只返回近邻数作为简化指标
                structures.append(('FCC-like', len(neighbors)))
            elif len(neighbors) == 14:
                structures.append(('BCC-like', len(neighbors)))
            else:
                structures.append(('Other', len(neighbors)))
        
        return structures
    
    def export_subset(self, output_file, condition):
        """导出满足条件的原子子集"""
        with open(output_file, 'w') as f:
            for frame in self.frames:
                selected = [a for a in frame['atoms'] if condition(a)]
                
                f.write(f"ITEM: TIMESTEP\n{frame['timestep']}\n")
                f.write(f"ITEM: NUMBER OF ATOMS\n{len(selected)}\n")
                f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
                for bound in frame['box']:
                    f.write(f"{bound[0]} {bound[1]}\n")
                f.write(f"ITEM: ATOMS id type x y z\n")
                for atom in selected:
                    f.write(f"{int(atom['id'])} {int(atom['type'])} "
                           f"{atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")
    
    def write_lammpstrj(self, output_file, start=0, end=None, stride=1):
        """写入标准lammpstrj格式"""
        if end is None:
            end = len(self.frames)
        
        with open(output_file, 'w') as f:
            for frame in self.frames[start:end:stride]:
                f.write(f"ITEM: TIMESTEP\n{frame['timestep']}\n")
                f.write(f"ITEM: NUMBER OF ATOMS\n{frame['natoms']}\n")
                f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
                for bound in frame['box']:
                    f.write(f"{bound[0]} {bound[1]}\n")
                f.write(f"ITEM: ATOMS id type x y z\n")
                for atom in frame['atoms']:
                    f.write(f"{int(atom['id'])} {int(atom['type'])} "
                           f"{atom['x']:.6f} {atom['y']:.6f} {atom['z']:.6f}\n")

def main():
    parser = argparse.ArgumentParser(description='Process LAMMPS trajectory')
    parser.add_argument('dump_file', help='LAMMPS dump file')
    parser.add_argument('--rdf', action='store_true', help='Compute RDF')
    parser.add_argument('--msd', action='store_true', help='Compute MSD')
    parser.add_argument('--clusters', action='store_true', help='Cluster analysis')
    parser.add_argument('--output', default='processed.dat', help='Output file')
    
    args = parser.parse_args()
    
    processor = TrajectoryProcessor(args.dump_file)
    
    if args.rdf:
        r, g_r = processor.compute_rdf()
        np.savetxt('rdf.dat', np.column_stack([r, g_r]))
        print(f"RDF saved to rdf.dat")
    
    if args.msd:
        timesteps, msd = processor.compute_msd()
        if timesteps is not None:
            np.savetxt('msd.dat', np.column_stack([timesteps, msd]))
            print(f"MSD saved to msd.dat")
    
    if args.clusters:
        clusters = processor.compute_cluster_analysis()
        sizes = [len(c) for c in clusters]
        print(f"Found {len(clusters)} clusters")
        print(f"Cluster size distribution: {sorted(sizes, reverse=True)[:10]}")

if __name__ == '__main__':
    main()
