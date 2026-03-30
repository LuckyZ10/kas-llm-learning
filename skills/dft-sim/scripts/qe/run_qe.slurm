#!/bin/bash
# Quantum ESPRESSO作业提交脚本 (SLURM)
# 用法: sbatch run_qe.slurm

#SBATCH --job-name=qe_job
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --output=qe_%j.out
#SBATCH --error=qe_%j.err

# 加载模块
module purge
module load gcc/11.2
module load openmpi/4.1.5

# 设置QE路径
PW_EXEC="/opt/qe/7.4/bin/pw.x"

# 设置并行环境
export OMP_NUM_THREADS=1

# 运行计算
srun $PW_EXEC -in pw.in > pw.out 2>&1

# 检查收敛
if grep -q "convergence has been achieved" pw.out; then
    echo "Calculation converged successfully!"
else
    echo "Warning: Calculation may not have converged!"
fi
