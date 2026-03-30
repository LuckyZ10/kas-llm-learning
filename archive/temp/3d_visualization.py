#!/usr/bin/env python3
"""
3D可视化完整示例
功能：PyVista体积渲染、Blender脚本、3D动画
"""

import numpy as np
from pathlib import Path

# PyVista
try:
    import pyvista as pv
    from pyvista import themes
    pv.set_plot_theme(themes.DocumentTheme())
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("[WARN] PyVista未安装")

# 其他3D库
try:
    import vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False


class Molecular3DVisualizer:
    """分子3D可视化器"""
    
    # CPK颜色
    CPK_COLORS = {
        1: (1.0, 1.0, 1.0),    # H - 白
        6: (0.4, 0.4, 0.4),    # C - 灰
        7: (0.0, 0.0, 1.0),    # N - 蓝
        8: (1.0, 0.0, 0.0),    # O - 红
        15: (1.0, 0.6, 0.0),   # P - 橙
        16: (1.0, 1.0, 0.0),   # S - 黄
        17: (0.0, 1.0, 0.0),   # Cl - 绿
        35: (0.6, 0.2, 0.0),   # Br - 棕
        79: (1.0, 0.8, 0.0),   # Au - 金
    }
    
    # 范德华半径 (Å)
    VDW_RADII = {
        1: 1.2, 6: 1.7, 7: 1.55, 8: 1.52,
        15: 1.8, 16: 1.8, 17: 1.75, 35: 1.85
    }
    
    def __init__(self):
        self.plotter = None
        self.actors = []
    
    def create_atom_sphere(self, position, element, resolution=32):
        """创建原子球体"""
        radius = self.VDW_RADII.get(element, 1.5)
        color = self.CPK_COLORS.get(element, (0.5, 0.5, 0.5))
        
        sphere = pv.Sphere(
            radius=radius,
            center=position,
            theta_resolution=resolution,
            phi_resolution=resolution
        )
        
        return sphere, color
    
    def create_bond_cylinder(self, pos1, pos2, radius=0.15, resolution=16):
        """创建化学键圆柱体"""
        # 计算中心点和方向
        center = (np.array(pos1) + np.array(pos2)) / 2
        direction = np.array(pos2) - np.array(pos1)
        length = np.linalg.norm(direction)
        direction = direction / length
        
        cylinder = pv.Cylinder(
            center=center,
            direction=direction,
            radius=radius,
            height=length,
            resolution=resolution
        )
        
        return cylinder
    
    def visualize_molecule(self, positions, elements, bonds=None, 
                          show_box=True, background='white'):
        """
        可视化分子
        
        Parameters:
        -----------
        positions : np.ndarray (N, 3)
            原子位置
        elements : list or np.ndarray
            原子序数列表
        bonds : list of tuples, optional
            键连接列表 [(i, j), ...]
        """
        if not HAS_PYVISTA:
            print("[ERROR] PyVista未安装")
            return
        
        self.plotter = pv.Plotter(window_size=[1920, 1080])
        self.plotter.set_background(background)
        
        # 添加原子
        print(f"[INFO] 添加 {len(positions)} 个原子...")
        for i, (pos, elem) in enumerate(zip(positions, elements)):
            sphere, color = self.create_atom_sphere(pos, elem)
            actor = self.plotter.add_mesh(
                sphere, 
                color=color,
                smooth_shading=True,
                specular=0.5,
                specular_power=20
            )
            self.actors.append(actor)
        
        # 添加化学键
        if bonds:
            print(f"[INFO] 添加 {len(bonds)} 个化学键...")
            for i, j in bonds:
                cylinder = self.create_bond_cylinder(positions[i], positions[j])
                actor = self.plotter.add_mesh(
                    cylinder,
                    color='gray',
                    smooth_shading=True
                )
                self.actors.append(actor)
        
        # 添加单元格
        if show_box:
            # 计算包围盒
            min_pos = np.min(positions, axis=0) - 2
            max_pos = np.max(positions, axis=0) + 2
            bounds = (*min_pos, *max_pos)
            
            box = pv.Box(bounds=bounds)
            self.plotter.add_mesh(box, style='wireframe', color='gray', opacity=0.3)
        
        # 添加坐标轴
        self.plotter.add_axes()
        
        # 添加标题
        self.plotter.add_text(f'Molecule: {len(positions)} atoms', 
                             position='upper_left', font_size=14)
        
        print("[OK] 分子可视化准备完成")
        return self.plotter
    
    def visualize_trajectory(self, trajectory, elements, interval=50):
        """
        可视化分子动力学轨迹动画
        
        Parameters:
        -----------
        trajectory : np.ndarray (N_frames, N_atoms, 3)
            轨迹数据
        elements : list
            原子序数
        interval : int
            帧间隔(ms)
        """
        if not HAS_PYVISTA:
            print("[ERROR] PyVista未安装")
            return
        
        self.plotter = pv.Plotter(window_size=[1920, 1080])
        self.plotter.set_background('white')
        
        # 初始位置
        positions = trajectory[0]
        
        # 创建初始球体
        spheres = []
        for pos, elem in zip(positions, elements):
            sphere = pv.Sphere(
                radius=self.VDW_RADII.get(elem, 1.5),
                center=pos,
                theta_resolution=24,
                phi_resolution=24
            )
            color = self.CPK_COLORS.get(elem, (0.5, 0.5, 0.5))
            actor = self.plotter.add_mesh(sphere, color=color, smooth_shading=True)
            spheres.append(actor)
        
        # 添加文本
        text = self.plotter.add_text('Frame: 0', position='upper_left', font_size=14)
        
        # 动画回调
        def update_frame(frame):
            # 这里需要更新所有球体的位置
            # 在PyVista中，我们需要重新创建网格
            pass
        
        print(f"[INFO] 轨迹包含 {len(trajectory)} 帧")
        print("[WARN] 轨迹动画需要手动实现帧更新逻辑")
        
        return self.plotter
    
    def create_movie(self, trajectory, elements, output_file='trajectory.mp4', fps=30):
        """创建轨迹视频"""
        if not HAS_PYVISTA:
            print("[ERROR] PyVista未安装")
            return
        
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        plotter.open_movie(output_file, framerate=fps)
        
        print(f"[INFO] 开始渲染视频: {output_file}")
        
        for i, positions in enumerate(trajectory):
            plotter.clear()
            
            # 添加所有原子
            for pos, elem in zip(positions, elements):
                sphere = pv.Sphere(
                    radius=self.VDW_RADII.get(elem, 1.5),
                    center=pos
                )
                color = self.CPK_COLORS.get(elem, (0.5, 0.5, 0.5))
                plotter.add_mesh(sphere, color=color, smooth_shading=True)
            
            # 添加帧标签
            plotter.add_text(f'Frame: {i}', position='upper_left', font_size=14)
            
            # 写入帧
            plotter.write_frame()
            
            if i % 10 == 0:
                print(f"  渲染进度: {i}/{len(trajectory)}")
        
        plotter.close()
        print(f"[OK] 视频保存到: {output_file}")
    
    def visualize_volume(self, scalar_field, origin=(0, 0, 0), 
                        spacing=(1, 1, 1), 
                        isosurfaces=None,
                        colormap='viridis'):
        """
        可视化体积数据（电子密度、电荷密度等）
        
        Parameters:
        -----------
        scalar_field : np.ndarray (nx, ny, nz)
            标量场数据
        origin : tuple
            原点坐标
        spacing : tuple
            网格间距
        isosurfaces : list
            等值面值列表
        """
        if not HAS_PYVISTA:
            print("[ERROR] PyVista未安装")
            return
        
        self.plotter = pv.Plotter(window_size=[1920, 1080])
        
        # 创建均匀网格
        grid = pv.UniformGrid()
        grid.dimensions = np.array(scalar_field.shape) + 1
        grid.origin = origin
        grid.spacing = spacing
        grid.cell_data['values'] = scalar_field.flatten(order='F')
        
        # 等值面
        if isosurfaces:
            for iso in isosurfaces:
                contour = grid.contour([iso])
                self.plotter.add_mesh(
                    contour, 
                    opacity=0.5,
                    colormap=colormap,
                    show_scalar_bar=True
                )
        
        # 体积渲染
        self.plotter.add_volume(
            grid,
            cmap=colormap,
            opacity='sigmoid',
            show_scalar_bar=True
        )
        
        self.plotter.add_axes()
        
        print(f"[OK] 体积数据可视化准备完成")
        return self.plotter
    
    def visualize_vector_field(self, positions, vectors, 
                              scale=1.0, 
                              colormap='viridis'):
        """
        可视化矢量场（速度场、力场等）
        
        Parameters:
        -----------
        positions : np.ndarray (N, 3)
            矢量起点位置
        vectors : np.ndarray (N, 3)
            矢量值
        """
        if not HAS_PYVISTA:
            print("[ERROR] PyVista未安装")
            return
        
        self.plotter = pv.Plotter(window_size=[1920, 1080])
        
        # 计算矢量大小
        magnitudes = np.linalg.norm(vectors, axis=1)
        
        # 创建箭头
        pdata = pv.PolyData(positions)
        pdata['vectors'] = vectors
        pdata['magnitudes'] = magnitudes
        
        # 箭头glyph
        arrows = pdata.glyph(
            orient='vectors',
            scale=True,
            factor=scale,
            clim=[magnitudes.min(), magnitudes.max()]
        )
        
        self.plotter.add_mesh(arrows, colormap=colormap, show_scalar_bar=True)
        
        # 添加矢量场流线
        grid = pv.UniformGrid()
        # 这里需要更复杂的网格插值来创建流线
        
        self.plotter.add_axes()
        
        print(f"[OK] 矢量场可视化准备完成")
        return self.plotter
    
    def create_slice_visualization(self, scalar_field, normal='z', 
                                   n_slices=5, colormap='viridis'):
        """
        创建切片可视化
        
        Parameters:
        -----------
        scalar_field : np.ndarray (nx, ny, nz)
            标量场
        normal : str
            切片方向 ('x', 'y', 'z')
        n_slices : int
            切片数量
        """
        if not HAS_PYVISTA:
            print("[ERROR] PyVista未安装")
            return
        
        self.plotter = pv.Plotter(window_size=[1920, 1080])
        
        # 创建网格
        grid = pv.UniformGrid()
        grid.dimensions = np.array(scalar_field.shape) + 1
        grid.cell_data['values'] = scalar_field.flatten(order='F')
        
        # 正交切片
        slices = grid.slice_orthogonal()
        self.plotter.add_mesh(slices, colormap=colormap, show_scalar_bar=True)
        
        # 添加等值面
        mean_val = np.mean(scalar_field)
        std_val = np.std(scalar_field)
        contour = grid.contour([mean_val + std_val, mean_val + 2*std_val])
        self.plotter.add_mesh(contour, opacity=0.3, colormap=colormap)
        
        self.plotter.add_axes()
        
        print(f"[OK] 切片可视化准备完成")
        return self.plotter
    
    def add_scalar_bar(self, title='Scalar', **kwargs):
        """添加标量条"""
        if self.plotter:
            self.plotter.add_scalar_bar(title=title, **kwargs)
    
    def save_screenshot(self, filename='screenshot.png', scale=2):
        """保存截图"""
        if self.plotter:
            self.plotter.screenshot(filename, scale=scale)
            print(f"[OK] 截图保存到: {filename}")
    
    def show(self):
        """显示图形"""
        if self.plotter:
            self.plotter.show()


class BlenderExporter:
    """Blender导出器 - 生成Blender Python脚本"""
    
    CPK_COLORS = {
        'H': (1.0, 1.0, 1.0),
        'C': (0.25, 0.25, 0.25),
        'N': (0.0, 0.0, 0.8),
        'O': (0.8, 0.0, 0.0),
        'P': (1.0, 0.6, 0.0),
        'S': (1.0, 0.9, 0.0),
    }
    
    VDW_RADII = {
        'H': 0.5, 'C': 0.7, 'N': 0.65, 'O': 0.6,
        'P': 0.9, 'S': 1.0
    }
    
    def generate_blender_script(self, positions, elements, bonds=None, 
                                output_file='molecule_blender.py'):
        """
        生成Blender Python脚本
        """
        script = '''#!/usr/bin/env python3
import bpy
import bmesh
import mathutils

# 清除场景
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 删除所有材质
for material in bpy.data.materials:
    bpy.data.materials.remove(material)

'''
        
        # 材质定义
        unique_elements = set(elements)
        for elem in unique_elements:
            color = self.CPK_COLORS.get(elem, (0.5, 0.5, 0.5))
            script += f'''
# 创建{elem}材质
mat_{elem} = bpy.data.materials.new(name="{elem}_material")
mat_{elem}.use_nodes = True
bsdf = mat_{elem}.node_tree.nodes["Principled BSDF"]
bsdf.inputs['Base Color'].default_value = ({color[0]}, {color[1]}, {color[2]}, 1.0)
bsdf.inputs['Roughness'].default_value = 0.2
bsdf.inputs['Metallic'].default_value = 0.1
'''
        
        # 创建原子
        script += '\n# 创建原子\n'
        for i, (pos, elem) in enumerate(zip(positions, elements)):
            radius = self.VDW_RADII.get(elem, 0.7)
            script += f'''
# 原子 {i} ({elem})
bpy.ops.mesh.primitive_uv_sphere_add(
    radius={radius},
    location=({pos[0]/10:.4f}, {pos[1]/10:.4f}, {pos[2]/10:.4f}),
    segments=32,
    ring_count=16
)
atom_{i} = bpy.context.active_object
atom_{i}.data.materials.append(mat_{elem})
'''
        
        # 创建化学键
        if bonds:
            script += '\n# 创建化学键\n'
            for i, (idx1, idx2) in enumerate(bonds):
                pos1 = positions[idx1]
                pos2 = positions[idx2]
                
                center = [(p1 + p2) / 20 for p1, p2 in zip(pos1, pos2)]  # 转换为Blender单位
                length = np.linalg.norm(np.array(pos2) - np.array(pos1)) / 10
                
                # 计算旋转
                direction = np.array(pos2) - np.array(pos1)
                direction = direction / np.linalg.norm(direction)
                
                script += f'''
# 键 {i}
bpy.ops.mesh.primitive_cylinder_add(
    radius=0.15,
    depth={length:.4f},
    location=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})
)
bond_{i} = bpy.context.active_object
'''
        
        # 添加灯光和相机
        script += '''
# 添加灯光
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
light = bpy.context.active_object
light.data.energy = 3

# 添加相机
bpy.ops.object.camera_add(location=(10, -10, 8))
camera = bpy.context.active_object
camera.rotation_euler = (1.1, 0, 0.785)
bpy.context.scene.camera = camera

# 渲染设置
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x = 3840
scene.render.resolution_y = 2160
scene.render.resolution_percentage = 100

print("Molecule imported successfully!")
print(f"Atoms: ''' + str(len(positions)) + '''")
'''
        
        # 保存脚本
        with open(output_file, 'w') as f:
            f.write(script)
        
        print(f"[OK] Blender脚本生成: {output_file}")
        print(f"使用方法: blender --python {output_file}")
    
    def generate_render_script(self, output_file='render.py'):
        """生成渲染脚本"""
        script = '''#!/usr/bin/env python3
import bpy

# 渲染设置
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 512
scene.render.resolution_x = 3840
scene.render.resolution_y = 2160
scene.render.resolution_percentage = 100

# 输出路径
scene.render.filepath = '//molecule_render.png'

# 渲染
bpy.ops.render.render(write_still=True)
print("Render complete!")
'''
        with open(output_file, 'w') as f:
            f.write(script)
        
        print(f"[OK] 渲染脚本生成: {output_file}")


# 使用示例
if __name__ == '__main__':
    print("=== 3D可视化模块 ===\n")
    
    # 创建示例分子数据（甲烷）
    print("1. 创建示例分子结构...")
    
    # 甲烷原子位置
    methane_positions = np.array([
        [0.0, 0.0, 0.0],      # C
        [1.09, 0.0, 0.0],     # H
        [-0.36, 1.03, 0.0],   # H
        [-0.36, -0.51, 0.89], # H
        [-0.36, -0.51, -0.89] # H
    ])
    methane_elements = [6, 1, 1, 1, 1]
    methane_bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
    
    # PyVista可视化
    if HAS_PYVISTA:
        print("\n2. PyVista可视化...")
        visualizer = Molecular3DVisualizer()
        plotter = visualizer.visualize_molecule(
            methane_positions, 
            methane_elements,
            bonds=methane_bonds
        )
        visualizer.save_screenshot('methane_pyvista.png')
        print("   截图保存: methane_pyvista.png")
        
        # 体积数据示例
        print("\n3. 体积数据可视化...")
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        z = np.linspace(-5, 5, 50)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # 高斯分布
        scalar_field = np.exp(-(X**2 + Y**2 + Z**2) / 10)
        
        plotter2 = visualizer.visualize_volume(
            scalar_field,
            isosurfaces=[0.3, 0.6],
            colormap='plasma'
        )
        visualizer.save_screenshot('volume_pyvista.png')
        print("   截图保存: volume_pyvista.png")
    
    # Blender导出
    print("\n4. Blender脚本生成...")
    blender = BlenderExporter()
    blender.generate_blender_script(
        methane_positions,
        ['C', 'H', 'H', 'H', 'H'],
        methane_bonds,
        'methane_blender.py'
    )
    blender.generate_render_script('render.py')
    
    print("\n=== 完成 ===")
    print("提示: 运行PyVista可视化需要显示器")
    print("提示: Blender脚本可以在无GUI环境下生成")
