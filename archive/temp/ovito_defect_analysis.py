#!/usr/bin/env python3
"""
OVITO缺陷识别完整示例
功能：位错分析、空位识别、晶体结构分类
"""

from ovito.io import import_file, export_file
from ovito.modifiers import *
from ovito.vis import *
from ovito.pipeline import Pipeline
import numpy as np

class OVITODefectAnalyzer:
    """OVITO缺陷分析器"""
    
    def __init__(self, dump_file, reference_file=None):
        self.pipeline = import_file(dump_file)
        self.reference_file = reference_file
        self.results = {}
        
    def identify_crystal_structure(self, structure_type='FCC'):
        """
        识别晶体结构
        structure_type: 'FCC', 'BCC', 'HCP', 'Diamond'
        """
        print(f"[INFO] 识别{structure_type}晶体结构...")
        
        # 常见邻居分析 (CNA)
        cna = CommonNeighborAnalysisModifier()
        cna.mode = CommonNeighborAnalysisModifier.Mode.FixedCutoff
        cna.cutoff = 3.5
        self.pipeline.modifiers.append(cna)
        
        # 根据结构类型选择分析器
        if structure_type == 'Diamond':
            diamond = IdentifyDiamondModifier()
            self.pipeline.modifiers.append(diamond)
        
        # 计算
        output = self.pipeline.compute()
        
        # 统计结构类型
        structure_types = output.particles['Structure Type']
        unique, counts = np.unique(structure_types, return_counts=True)
        
        structure_names = {0: 'Other', 1: 'FCC', 2: 'HCP', 3: 'BCC', 4: 'ICO'}
        
        self.results['crystal_structure'] = {
            structure_names.get(u, f'Type_{u}'): int(c) 
            for u, c in zip(unique, counts)
        }
        
        print(f"[OK] 晶体结构分析完成:")
        for name, count in self.results['crystal_structure'].items():
            print(f"  - {name}: {count} atoms")
            
        return self.results['crystal_structure']
    
    def analyze_dislocations(self, crystal_type='FCC'):
        """
        位错分析 (DXA)
        crystal_type: 'FCC', 'BCC', 'Diamond'
        """
        print(f"[INFO] 执行位错分析 (DXA)...")
        
        # 位错分析器
        dxa = DislocationAnalysisModifier()
        
        # 设置晶体类型
        type_map = {
            'FCC': DislocationAnalysisModifier.CrystalType.FCC,
            'BCC': DislocationAnalysisModifier.CrystalType.BCC,
            'Diamond': DislocationAnalysisModifier.CrystalType.CubicDiamond,
            'DiamondHex': DislocationAnalysisModifier.CrystalType.HexagonalDiamond
        }
        dxa.input_crystal_structure = type_map.get(crystal_type, DislocationAnalysisModifier.CrystalType.FCC)
        
        self.pipeline.modifiers.append(dxa)
        
        # 计算
        output = self.pipeline.compute()
        
        # 提取位错信息
        disloc_network = output.data.dislocations
        
        self.results['dislocations'] = {
            'segments': len(disloc_network.segments) if disloc_network else 0,
            'total_line_length': float(disloc_network.total_line_length) if disloc_network else 0,
            'burgers_vectors': []
        }
        
        if disloc_network:
            for segment in disloc_network.segments:
                self.results['dislocations']['burgers_vectors'].append(
                    segment.true_burgers_vector.tolist()
                )
        
        print(f"[OK] 位错分析完成:")
        print(f"  - 位错线段数: {self.results['dislocations']['segments']}")
        print(f"  - 总位错线长度: {self.results['dislocations']['total_line_length']:.2f} Å")
        
        return self.results['dislocations']
    
    def analyze_defects_wigner_seitz(self):
        """
        Wigner-Seitz缺陷分析 - 识别空位和间隙原子
        """
        if not self.reference_file:
            print("[ERROR] 需要提供参考晶体结构文件")
            return None
            
        print(f"[INFO] 执行Wigner-Seitz缺陷分析...")
        
        # 加载参考结构
        ref_pipeline = import_file(self.reference_file)
        
        # Wigner-Seitz分析
        ws = WignerSeitzAnalysisModifier()
        ws.reference = ref_pipeline.source
        ws.eliminate_cell_deformation = True
        
        self.pipeline.modifiers.append(ws)
        
        # 计算
        output = self.pipeline.compute()
        
        # 提取缺陷信息
        vacancies = output.attributes.get('WignerSeitz.vacancy_count', 0)
        interstitials = output.attributes.get('WignerSeitz.interstitial_count', 0)
        antisites = output.attributes.get('WignerSeitz.antisite_count', 0)
        
        # 获取缺陷位置
        occupancy = output.particles['Occupancy']
        positions = output.particles.positions
        
        vacancy_positions = positions[occupancy == 0]
        interstitial_positions = positions[occupancy > 1]
        
        self.results['wigner_seitz'] = {
            'vacancies': int(vacancies),
            'interstitials': int(interstitials),
            'antisites': int(antisites),
            'vacancy_positions': vacancy_positions.tolist(),
            'interstitial_positions': interstitial_positions.tolist()
        }
        
        print(f"[OK] Wigner-Seitz分析完成:")
        print(f"  - 空位数: {vacancies}")
        print(f"  - 间隙原子数: {interstitials}")
        print(f"  - 反位缺陷数: {antisites}")
        
        return self.results['wigner_seitz']
    
    def voronoi_analysis(self):
        """Voronoi单元分析 - 局部原子环境"""
        print(f"[INFO] 执行Voronoi分析...")
        
        voronoi = VoronoiAnalysisModifier(
            compute_indices=True,
            use_radii=False,
            edge_threshold=0.1
        )
        self.pipeline.modifiers.append(voronoi)
        
        output = self.pipeline.compute()
        
        # Voronoi体积
        volumes = output.particles['Atomic Volume']
        
        # Voronoi指数
        indices = output.particles['Voronoi Index']
        
        # 统计常见指数
        index_counts = {}
        for idx in indices:
            key = tuple(idx[:6])  # 取前6个指数
            index_counts[key] = index_counts.get(key, 0) + 1
        
        # 按频率排序
        sorted_indices = sorted(index_counts.items(), key=lambda x: x[1], reverse=True)
        
        self.results['voronoi'] = {
            'mean_volume': float(np.mean(volumes)),
            'std_volume': float(np.std(volumes)),
            'top_indices': sorted_indices[:10]
        }
        
        print(f"[OK] Voronoi分析完成:")
        print(f"  - 平均原子体积: {self.results['voronoi']['mean_volume']:.2f} Å³")
        print(f"  - 常见Voronoi指数 (前5):")
        for idx, count in sorted_indices[:5]:
            print(f"    <{idx}>: {count} atoms")
        
        return self.results['voronoi']
    
    def coordination_analysis(self, cutoff=3.5):
        """配位数分析"""
        print(f"[INFO] 执行配位数分析 (cutoff={cutoff}Å)...")
        
        coord = CoordinationAnalysisModifier(
            cutoff=cutoff,
            number_of_bins=100
        )
        self.pipeline.modifiers.append(coord)
        
        output = self.pipeline.compute()
        
        # 配位数
        coordination = output.particles['Coordination']
        
        # 统计配位分布
        coord_counts = {}
        for c in coordination:
            coord_counts[c] = coord_counts.get(c, 0) + 1
        
        self.results['coordination'] = {
            'mean_coordination': float(np.mean(coordination)),
            'distribution': coord_counts
        }
        
        print(f"[OK] 配位数分析完成:")
        print(f"  - 平均配位数: {self.results['coordination']['mean_coordination']:.2f}")
        print(f"  - 配位分布: {dict(sorted(coord_counts.items()))}")
        
        return self.results['coordination']
    
    def cluster_analysis(self, cutoff=3.5):
        """团簇分析"""
        print(f"[INFO] 执行团簇分析...")
        
        cluster = ClusterAnalysisModifier(
            cutoff=cutoff,
            sort_by_size=True
        )
        self.pipeline.modifiers.append(cluster)
        
        output = self.pipeline.compute()
        
        # 团簇信息
        cluster_ids = output.particles['Cluster']
        
        # 统计团簇大小
        unique, counts = np.unique(cluster_ids, return_counts=True)
        
        self.results['clusters'] = {
            'num_clusters': len(unique) - (1 if 0 in unique else 0),
            'largest_cluster_size': int(max(counts)),
            'cluster_sizes': counts.tolist()
        }
        
        print(f"[OK] 团簇分析完成:")
        print(f"  - 团簇数量: {self.results['clusters']['num_clusters']}")
        print(f"  - 最大团簇大小: {self.results['clusters']['largest_cluster_size']}")
        
        return self.results['clusters']
    
    def export_defect_visualization(self, output_file='defects.xyz'):
        """导出缺陷可视化文件"""
        print(f"[INFO] 导出可视化文件到 {output_file}...")
        
        export_file(
            self.pipeline, 
            output_file, 
            'xyz',
            columns=[
                'Particle Identifier',
                'Particle Type',
                'Position.X', 'Position.Y', 'Position.Z',
                'Structure Type',
                'Coordination'
            ]
        )
        
        print(f"[OK] 导出完成")
    
    def generate_report(self):
        """生成分析报告"""
        report = """
========================================
OVITO 缺陷分析报告
========================================

1. 晶体结构分析:
"""
        if 'crystal_structure' in self.results:
            for name, count in self.results['crystal_structure'].items():
                report += f"   - {name}: {count} atoms\n"
        
        report += "\n2. 位错分析:\n"
        if 'dislocations' in self.results:
            d = self.results['dislocations']
            report += f"   - 位错线段数: {d['segments']}\n"
            report += f"   - 总位错线长度: {d['total_line_length']:.2f} Å\n"
        
        report += "\n3. 点缺陷分析:\n"
        if 'wigner_seitz' in self.results:
            d = self.results['wigner_seitz']
            report += f"   - 空位数: {d['vacancies']}\n"
            report += f"   - 间隙原子数: {d['interstitials']}\n"
            report += f"   - 反位缺陷数: {d['antisites']}\n"
        
        report += "\n4. 局部结构分析:\n"
        if 'voronoi' in self.results:
            v = self.results['voronoi']
            report += f"   - 平均原子体积: {v['mean_volume']:.2f} Å³\n"
        if 'coordination' in self.results:
            c = self.results['coordination']
            report += f"   - 平均配位数: {c['mean_coordination']:.2f}\n"
        
        report += "\n========================================\n"
        
        print(report)
        return report


# 使用示例
if __name__ == '__main__':
    # 创建分析器
    analyzer = OVITODefectAnalyzer(
        dump_file='trajectory.dump',
        reference_file='reference.crystal'
    )
    
    # 运行分析
    analyzer.identify_crystal_structure('FCC')
    analyzer.analyze_dislocations('FCC')
    analyzer.analyze_defects_wigner_seitz()
    analyzer.voronoi_analysis()
    analyzer.coordination_analysis()
    analyzer.cluster_analysis()
    
    # 生成报告
    analyzer.generate_report()
    
    # 导出可视化
    analyzer.export_defect_visualization('defects.xyz')
