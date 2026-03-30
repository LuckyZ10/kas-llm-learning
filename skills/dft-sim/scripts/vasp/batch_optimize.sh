#!/bin/bash
# 批量结构优化脚本
# 遍历子目录中的所有结构文件并进行优化

# 设置
VASP_EXEC="vasp_std"
MAX_JOBS=4          # 最大并行作业数
CURRENT_JOBS=0

# 函数: 运行VASP优化
run_optimization() {
    local dir=$1
    cd $dir
    
    echo "Processing: $dir"
    
    # 检查必要文件
    if [ ! -f POSCAR ]; then
        echo "  Error: POSCAR not found in $dir"
        cd ..
        return 1
    fi
    
    # 创建优化INCAR (如果不存在)
    if [ ! -f INCAR ]; then
        cat > INCAR << EOF
SYSTEM = Structure Optimization
ENCUT = 520
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-6
EDIFFG = -0.01
IBRION = 2
ISIF = 3
NSW = 100
POTIM = 0.5
NCORE = 4
EOF
        echo "  Created default INCAR"
    fi
    
    # 创建KPOINTS (如果不存在)
    if [ ! -f KPOINTS ]; then
        cat > KPOINTS << EOF
Automatic mesh
0
Gamma
6 6 6
0 0 0
EOF
        echo "  Created default KPOINTS"
    fi
    
    # 运行优化
    mpirun -np 4 $VASP_EXEC > vasp.out 2>&1
    
    # 检查结果
    if grep -q "reached required accuracy" OUTCAR; then
        echo "  ✓ Converged"
        # 保存最终结构
        cp CONTCAR ${dir}_optimized.vasp
    else
        echo "  ✗ Not converged"
    fi
    
    cd ..
}

# 主循环
echo "Starting batch structure optimization..."
echo "======================================="

for dir in */; do
    # 跳过非目录项
    [ -d "$dir" ] || continue
    
    # 跳过当前目录和父目录
    [ "$dir" = "./" ] && continue
    [ "$dir" = "../" ] && continue
    
    # 运行优化
    run_optimization "$dir" &
    
    # 控制并行度
    CURRENT_JOBS=$((CURRENT_JOBS + 1))
    if [ $CURRENT_JOBS -ge $MAX_JOBS ]; then
        wait
        CURRENT_JOBS=0
    fi
done

# 等待所有后台作业完成
wait

echo ""
echo "======================================="
echo "Batch optimization completed!"
echo "Optimized structures saved as *_optimized.vasp"
