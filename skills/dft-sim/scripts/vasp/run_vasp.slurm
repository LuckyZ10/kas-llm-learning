#!/bin/bash
# VASP作业提交脚本模板 (SLURM)
# 用法: sbatch run_vasp.slurm

#SBATCH --job-name=vasp_job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --output=vasp_%j.out
#SBATCH --error=vasp_%j.err

# 加载模块
module purge
module load intel/2023.1
module load intel-mpi/2021.9

# 设置VASP路径
VASP_EXEC="/opt/vasp/6.5.0/bin/vasp_std"

# 设置并行环境
export OMP_NUM_THREADS=1
export I_MPI_FABRICS=shm:ofi

# 运行VASP
srun $VASP_EXEC

# 检查收敛
if grep -q "reached required accuracy" OUTCAR; then
    echo "Calculation converged successfully!"
else
    echo "Warning: Calculation may not have converged!"
fi
