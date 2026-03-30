#!/usr/bin/env python3
"""
examples/batch_submission.py
===========================
批量作业提交示例

展示如何批量提交计算作业，适用于高通量计算场景。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from hpc_integration.job_submitter import (
    JobTemplate, JobArrayBuilder, CalculationType
)


def create_batch_submission_config():
    """创建批量提交配置"""
    return {
        "project_name": "battery_materials",
        "structures_dir": "./structures",
        "calculation_type": "vasp",
        "batch_size": 100,
        "resources": {
            "nodes": 1,
            "cores_per_node": 16,
            "memory_gb": 64,
            "walltime_hours": 12.0,
            "queue": "normal"
        },
        "output_dir": "./batch_results"
    }


def generate_structure_list(count: int = 50):
    """生成模拟的结构列表"""
    materials = [
        "Li", "Na", "K", "Mg", "Ca", "Al", "Si", "P", "S", "Cl"
    ]
    structures = []
    
    for i in range(count):
        mat = materials[i % len(materials)]
        structures.append({
            "id": i,
            "name": f"{mat}_structure_{i:04d}",
            "file": f"POSCAR_{mat}_{i:04d}",
            "expected_energy": None
        })
    
    return structures


def submit_batch_calculations(config: dict, structures: list):
    """批量提交计算"""
    print(f"批量提交: {len(structures)} 个结构")
    print(f"项目: {config['project_name']}")
    print(f"计算类型: {config['calculation_type']}")
    
    # 创建作业模板
    template = JobTemplate.for_vasp(
        name=f"{config['project_name']}_calc",
        **config['resources']
    )
    
    # 创建作业数组构建器
    work_dir = Path(config['output_dir'])
    work_dir.mkdir(parents=True, exist_ok=True)
    
    builder = JobArrayBuilder(template, work_dir)
    
    # 批量添加作业
    for struct in structures:
        builder.add_job(
            working_dir=work_dir / f"calc_{struct['id']:04d}",
            job_name=f"calc_{struct['name']}",
            custom_inputs={
                'pre_commands': [
                    f"echo 'Calculating {struct['name']}'",
                    f"cp {config['structures_dir']}/{struct['file']} POSCAR"
                ]
            }
        )
    
    # 保存提交信息
    submission_info = {
        "timestamp": datetime.now().isoformat(),
        "project": config['project_name'],
        "total_jobs": len(structures),
        "work_dir": str(work_dir),
        "template": template.to_dict()
    }
    
    info_file = work_dir / "submission_info.json"
    with open(info_file, 'w') as f:
        json.dump(submission_info, f, indent=2)
    
    print(f"\n提交信息已保存: {info_file}")
    print(f"工作目录: {work_dir}")
    print(f"作业数: {len(builder)}")
    
    # 返回数组作业规格（可用于实际提交）
    array_spec = builder.build_array_job()
    
    return array_spec, builder


def submit_with_dependencies(config: dict, structures: list):
    """提交带依赖的批量计算（流水线模式）"""
    print(f"\n流水线模式提交")
    
    work_dir = Path(config['output_dir']) / "pipeline"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # 阶段1: 结构松弛
    relax_template = JobTemplate.for_vasp(
        name="relax",
        nodes=1,
        cores_per_node=16,
        walltime_hours=8.0
    )
    
    # 阶段2: 静态计算（依赖于松弛）
    static_template = JobTemplate.for_vasp(
        name="static",
        nodes=1,
        cores_per_node=16,
        walltime_hours=4.0,
        executable="vasp_std"
    )
    static_template.arguments = ["# 静态计算参数"]
    
    # 阶段3: 后处理（依赖于静态）
    post_template = JobTemplate(
        name="postprocess",
        calculation_type=CalculationType.CUSTOM,
        nodes=1,
        cores_per_node=4,
        memory_gb=16,
        walltime_hours=1.0,
        executable="python",
        arguments=["analyze.py"]
    )
    
    print("阶段1: 结构松弛")
    print("阶段2: 静态计算 (依赖于阶段1)")
    print("阶段3: 后处理 (依赖于阶段2)")
    
    # 创建依赖链
    jobs = []
    for struct in structures[:5]:  # 限制数量用于演示
        job_chain = {
            "structure": struct['name'],
            "stages": ["relax", "static", "postprocess"]
        }
        jobs.append(job_chain)
    
    return jobs


def monitor_batch_progress(job_id: str, expected_count: int):
    """监控批量作业进度"""
    print(f"\n监控作业: {job_id}")
    print(f"预期完成: {expected_count} 个任务")
    
    # 模拟进度监控
    import time
    
    progress = {
        "pending": expected_count,
        "running": 0,
        "completed": 0,
        "failed": 0
    }
    
    for step in range(5):
        # 模拟状态变化
        if step > 0:
            progress["pending"] -= 10
            progress["running"] = 15
            progress["completed"] += 10
        
        completed_pct = (progress["completed"] / expected_count) * 100
        
        print(f"  Step {step + 1}: {completed_pct:.1f}% 完成")
        print(f"    待处理: {progress['pending']}")
        print(f"    运行中: {progress['running']}")
        print(f"    已完成: {progress['completed']}")
        
        time.sleep(0.1)  # 模拟延迟
    
    return progress


def collect_batch_results(output_dir: Path):
    """收集批量计算结果"""
    print(f"\n收集结果: {output_dir}")
    
    results = {
        "total": 0,
        "successful": 0,
        "failed": 0,
        "data": []
    }
    
    # 模拟结果收集
    for i in range(50):
        result = {
            "id": i,
            "status": "success" if i % 10 != 0 else "failed",
            "energy": -100.0 - i * 0.1 if i % 10 != 0 else None
        }
        
        results["total"] += 1
        if result["status"] == "success":
            results["successful"] += 1
        else:
            results["failed"] += 1
        
        results["data"].append(result)
    
    # 保存结果
    results_file = output_dir / "batch_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存: {results_file}")
    print(f"总计: {results['total']}")
    print(f"成功: {results['successful']}")
    print(f"失败: {results['failed']}")
    
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("HPC批量作业提交示例")
    print("=" * 60)
    
    # 创建配置
    config = create_batch_submission_config()
    
    # 生成结构列表
    structures = generate_structure_list(50)
    
    # 批量提交
    array_spec, builder = submit_batch_calculations(config, structures)
    
    # 流水线模式
    pipeline_jobs = submit_with_dependencies(config, structures)
    
    # 模拟监控
    monitor_batch_progress("12345", len(structures))
    
    # 收集结果
    output_dir = Path(config['output_dir'])
    results = collect_batch_results(output_dir)
    
    print("\n" + "=" * 60)
    print("批量提交示例完成!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
