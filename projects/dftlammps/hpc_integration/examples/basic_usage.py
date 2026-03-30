#!/usr/bin/env python3
"""
examples/basic_usage.py
=======================
HPC集成基础用法示例

展示如何使用HPC集成客户端提交作业、监控资源和管理数据。
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from hpc_integration import create_hpc_client
from hpc_integration.job_submitter import JobTemplate, JobArrayBuilder
from hpc_integration.cluster_connector import ClusterConfig


def example_1_basic_connection():
    """示例1: 基本连接"""
    print("=" * 60)
    print("示例1: 基本连接")
    print("=" * 60)
    
    # 配置集群
    config = ClusterConfig(
        name="hpc_cluster",
        host="hpc.example.com",
        username="user",
        key_file="~/.ssh/id_rsa",
        scheduler_type="slurm",
        remote_work_dir="~/dft_calculations"
    )
    
    # 创建客户端并连接
    client = create_hpc_client(
        cluster_config=config.to_dict(),
        enable_monitoring=True,
        enable_fault_tolerance=True,
        enable_checkpointing=True
    )
    
    print("客户端配置完成")
    print(f"集群: {config.name}")
    print(f"主机: {config.host}")
    print(f"调度器: {config.scheduler_type}")
    
    # 注意: 实际连接需要真实的HPC环境
    # with client:
    #     print("已连接到集群")
    
    return client


def example_2_submit_single_job():
    """示例2: 提交单个VASP作业"""
    print("\n" + "=" * 60)
    print("示例2: 提交单个VASP作业")
    print("=" * 60)
    
    # 创建作业模板
    template = JobTemplate.for_vasp(
        name="vasp_relaxation",
        nodes=2,
        cores_per_node=32,
        memory_gb=128,
        walltime_hours=24.0,
        queue="normal"
    )
    
    print(f"作业名称: {template.name}")
    print(f"计算类型: {template.calculation_type.value}")
    print(f"节点数: {template.nodes}")
    print(f"每节点核心数: {template.cores_per_node}")
    print(f"总核心数: {template.nodes * template.cores_per_node}")
    print(f"内存: {template.memory_gb} GB")
    print(f"运行时间: {template.walltime_hours} 小时")
    
    # 生成作业脚本
    from hpc_integration.job_submitter import EnhancedJobSubmitter
    
    class MockConnector:
        def mkdir(self, path, parents=True):
            print(f"创建目录: {path}")
    
    submitter = EnhancedJobSubmitter(MockConnector(), "slurm")
    
    from hpc_integration.job_submitter import JobSpec
    spec = JobSpec(
        template=template,
        working_dir=Path("./vasp_calc")
    )
    
    script = submitter._generate_script(spec)
    print("\n生成的Slurm脚本:")
    print("-" * 40)
    print(script)
    
    return template


def example_3_submit_job_array():
    """示例3: 提交作业数组（批量计算）"""
    print("\n" + "=" * 60)
    print("示例3: 提交作业数组（批量计算）")
    print("=" * 60)
    
    # 创建模板
    template = JobTemplate.for_vasp(
        name="batch_calc",
        nodes=1,
        cores_per_node=16,
        walltime_hours=12.0
    )
    
    # 创建作业数组构建器
    work_dir = Path("./batch_calculations")
    builder = JobArrayBuilder(template, work_dir)
    
    # 添加多个结构计算
    structures = [
        "POSCAR_Li",
        "POSCAR_Na",
        "POSCAR_K",
        "POSCAR_Mg",
        "POSCAR_Ca"
    ]
    
    for i, structure in enumerate(structures):
        builder.add_job(
            working_dir=work_dir / f"calc_{i:03d}",
            job_name=f"{structure}_relax",
            custom_inputs={
                'pre_commands': [f"cp ../{structure} POSCAR"]
            }
        )
    
    print(f"创建了 {len(builder)} 个作业")
    print(f"工作目录: {work_dir}")
    
    # 构建数组作业规格
    array_spec = builder.build_array_job()
    print(f"\n数组作业规格:")
    print(f"  名称: {array_spec.job_name}")
    print(f"  范围: {array_spec.array_range}")
    
    return builder


def example_4_ml_training_job():
    """示例4: ML势训练作业"""
    print("\n" + "=" * 60)
    print("示例4: ML势训练作业")
    print("=" * 60)
    
    # NEP训练
    nep_template = JobTemplate.for_nep(
        name="nep_training",
        nodes=1,
        cores_per_node=8,
        gpus=1,
        gpu_type="a100",
        memory_gb=128,
        walltime_hours=168.0  # 7天
    )
    
    print(f"训练类型: NEP")
    print(f"GPU数量: {nep_template.gpus}")
    print(f"GPU类型: {nep_template.gpu_type}")
    print(f"运行时间: {nep_template.walltime_hours} 小时")
    
    # DeepMD训练
    dp_template = JobTemplate.for_ml_training(
        name="deepmd_training",
        num_gpus=2,
        gpu_type="a100",
        walltime_hours=72.0
    )
    
    print(f"\n训练类型: DeepMD")
    print(f"GPU数量: {dp_template.gpus}")
    print(f"运行时间: {dp_template.walltime_hours} 小时")
    
    return nep_template, dp_template


def example_5_workflow_with_dependencies():
    """示例5: 带依赖的工作流"""
    print("\n" + "=" * 60)
    print("示例5: 带依赖的工作流")
    print("=" * 60)
    
    # 阶段1: DFT计算
    dft_template = JobTemplate.for_vasp(
        name="dft_stage",
        nodes=2,
        cores_per_node=32,
        walltime_hours=48.0
    )
    
    # 阶段2: 数据处理
    process_template = JobTemplate(
        name="process_stage",
        calculation_type=JobTemplate.from_dict({'name': 'custom', 'calculation_type': 'custom'}).calculation_type,
        nodes=1,
        cores_per_node=4,
        memory_gb=32,
        walltime_hours=2.0,
        executable="python",
        arguments=["process_results.py"]
    )
    
    # 阶段3: ML训练（依赖于数据处理）
    train_template = JobTemplate.for_ml_training(
        name="train_stage",
        num_gpus=1,
        walltime_hours=24.0
    )
    
    print("工作流阶段:")
    print(f"  1. DFT计算: {dft_template.name}")
    print(f"     节点: {dft_template.nodes}, 核心: {dft_template.cores_per_node}")
    print(f"  2. 数据处理: {process_template.name}")
    print(f"     依赖于: DFT计算")
    print(f"  3. ML训练: {train_template.name}")
    print(f"     依赖于: 数据处理")
    
    return [dft_template, process_template, train_template]


def example_6_container_execution():
    """示例6: 容器执行"""
    print("\n" + "=" * 60)
    print("示例6: 容器执行")
    print("=" * 60)
    
    from hpc_integration.container_runtime import (
        ContainerConfig, ContainerImage, ContainerEngine
    )
    
    # 配置容器
    config = ContainerConfig(
        image=ContainerImage(
            name="vasp",
            tag="6.4.0",
            registry="docker.io/hpc"
        ),
        engine=ContainerEngine.SINGULARITY,
        bind_mounts={
            "/home/user/data": "/data",
            "/scratch": "/tmp"
        },
        gpus=1,
        environment={
            "OMP_NUM_THREADS": "4"
        }
    )
    
    print(f"容器引擎: {config.engine.value}")
    print(f"镜像: {config.image.full_name}")
    print(f"绑定挂载: {config.bind_mounts}")
    print(f"GPU: {config.gpus}")
    
    return config


def example_7_fault_tolerance():
    """示例7: 容错与重试"""
    print("\n" + "=" * 60)
    print("示例7: 容错与重试")
    print("=" * 60)
    
    from hpc_integration.fault_tolerance import (
        FaultToleranceManager, RetryPolicy, FailureType
    )
    
    # 创建容错管理器
    ft_manager = FaultToleranceManager()
    
    # 配置重试策略
    retry_policy = RetryPolicy(
        max_attempts=5,
        initial_delay=1.0,
        backoff_factor=2.0,
        backoff_strategy="exponential"
    )
    
    print(f"重试策略:")
    print(f"  最大重试次数: {retry_policy.max_attempts}")
    print(f"  初始延迟: {retry_policy.initial_delay} 秒")
    print(f"  退避因子: {retry_policy.backoff_factor}")
    print(f"  策略: {retry_policy.backoff_strategy}")
    
    # 计算各次重试的延迟
    print("\n预期的重试延迟:")
    for attempt in range(retry_policy.max_attempts):
        delay = retry_policy.calculate_delay(attempt)
        print(f"  第 {attempt + 1} 次: {delay:.2f} 秒")
    
    # 获取断路器
    cb = ft_manager.get_circuit_breaker(
        name="job_submission",
        failure_threshold=5,
        recovery_timeout=60.0
    )
    
    print(f"\n断路器状态: {cb.state.value}")
    print(f"故障阈值: {cb.failure_threshold}")
    print(f"恢复超时: {cb.recovery_timeout} 秒")
    
    return ft_manager


def example_8_checkpoint_resume():
    """示例8: 检查点与恢复"""
    print("\n" + "=" * 60)
    print("示例8: 检查点与恢复")
    print("=" * 60)
    
    from hpc_integration.checkpoint_manager import (
        CheckpointManager, LocalCheckpointStorage, CalculationState
    )
    
    # 创建检查点管理器
    storage = LocalCheckpointStorage("./checkpoints")
    manager = CheckpointManager(
        storage=storage,
        auto_checkpoint_interval=3600,
        max_checkpoints=5
    )
    
    print(f"检查点目录: ./checkpoints")
    print(f"自动检查点间隔: 3600 秒")
    print(f"最大检查点数: 5")
    
    # 模拟创建检查点
    work_dir = Path("./simulation")
    work_dir.mkdir(exist_ok=True)
    
    checkpoint = manager.create_checkpoint(
        job_id="vasp_001",
        working_dir=work_dir,
        state=CalculationState.RUNNING,
        iteration=100,
        total_steps=1000,
        completed_steps=250,
        metadata={
            "system": "LiCoO2",
            "kpoints": "4x4x4"
        }
    )
    
    if checkpoint:
        print(f"\n检查点已创建:")
        print(f"  ID: {checkpoint.checkpoint_id}")
        print(f"  作业: {checkpoint.job_id}")
        print(f"  状态: {checkpoint.state.value}")
        print(f"  进度: {checkpoint.progress_pct:.1f}%")
        print(f"  时间: {checkpoint.timestamp}")
    
    # 分析恢复能力
    capability = manager.analyze_resume_capability("vasp_001", work_dir)
    
    print(f"\n恢复能力分析:")
    print(f"  可以恢复: {capability.can_resume}")
    if capability.resume_from:
        print(f"  从检查点: {capability.resume_from.checkpoint_id}")
    
    return manager


def example_9_storage_backend():
    """示例9: 存储后端"""
    print("\n" + "=" * 60)
    print("示例9: 存储后端")
    print("=" * 60)
    
    from hpc_integration.storage_backend import StorageConfig
    
    # S3配置
    s3_config = StorageConfig(
        type="s3",
        endpoint="https://s3.amazonaws.com",
        access_key="AKIAIOSFODNN7EXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        bucket="dft-lammps-data",
        region="us-east-1"
    )
    
    print("S3配置:")
    print(f"  端点: {s3_config.endpoint}")
    print(f"  存储桶: {s3_config.bucket}")
    print(f"  区域: {s3_config.region}")
    
    # MinIO配置
    minio_config = StorageConfig(
        type="minio",
        endpoint="http://localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        bucket="dft-data"
    )
    
    print("\nMinIO配置:")
    print(f"  端点: {minio_config.endpoint}")
    print(f"  存储桶: {minio_config.bucket}")
    
    # SFTP配置
    sftp_config = StorageConfig(
        type="sftp",
        host="data.hpc.example.com",
        port=22,
        username="user",
        key_file="~/.ssh/id_rsa"
    )
    
    print("\nSFTP配置:")
    print(f"  主机: {sftp_config.host}")
    print(f"  端口: {sftp_config.port}")
    
    return s3_config, minio_config, sftp_config


def main():
    """主函数"""
    print("=" * 60)
    print("HPC集成模块 - 基础用法示例")
    print("=" * 60)
    
    # 运行所有示例
    example_1_basic_connection()
    example_2_submit_single_job()
    example_3_submit_job_array()
    example_4_ml_training_job()
    example_5_workflow_with_dependencies()
    example_6_container_execution()
    example_7_fault_tolerance()
    example_8_checkpoint_resume()
    example_9_storage_backend()
    
    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
