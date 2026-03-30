#!/bin/bash
# 收敛性测试脚本
# 测试截断能和k点网格的收敛性

# 设置变量
PSEUDO_DIR="./pseudo"
STRUCT_FILE="POSCAR"
TEST_DIR="convergence_test"

# 创建测试目录
mkdir -p $TEST_DIR
cd $TEST_DIR

# 1. 截断能收敛测试
echo "=== ENCUT Convergence Test ==="
for ENCUT in 300 400 500 600 700 800; do
    mkdir -p encut_$ENCUT
    cd encut_$ENCUT
    
    # 创建INCAR
    cat > INCAR << EOF
SYSTEM = Convergence Test ENCUT=$ENCUT
ENCUT = $ENCUT
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-8
EOF
    
    # 复制必要文件
    cp ../$STRUCT_FILE POSCAR 2>/dev/null || echo "Warning: POSCAR not found"
    cp $PSEUDO_DIR/POTCAR . 2>/dev/null || echo "Warning: POTCAR not found"
    
    # 创建KPOINTS
    cat > KPOINTS << EOF
Automatic mesh
0
Gamma
6 6 6
0 0 0
EOF
    
    echo "Testing ENCUT = $ENCUT eV"
    # 运行VASP (注释掉，根据需要启用)
    # mpirun -np 4 vasp_std
    
    cd ..
done

# 2. k点网格收敛测试
echo ""
echo "=== KPOINTS Convergence Test ==="
for K in 4 6 8 10 12; do
    mkdir -p kpoints_${K}x${K}x${K}
    cd kpoints_${K}x${K}x${K}
    
    # 创建INCAR
    cat > INCAR << EOF
SYSTEM = Convergence Test K=$K
ENCUT = 520
ISMEAR = 0
SIGMA = 0.05
EDIFF = 1E-8
EOF
    
    # 复制必要文件
    cp ../$STRUCT_FILE POSCAR 2>/dev/null
    cp $PSEUDO_DIR/POTCAR . 2>/dev/null
    
    # 创建KPOINTS
    cat > KPOINTS << EOF
Automatic mesh
0
Gamma
$K $K $K
0 0 0
EOF
    
    echo "Testing KPOINTS = ${K}x${K}x${K}"
    # 运行VASP (注释掉，根据需要启用)
    # mpirun -np 4 vasp_std
    
    cd ..
done

# 3. 提取结果
echo ""
echo "=== Results Summary ==="
echo "ENCUT (eV) | Energy (eV) | Time (s)"
echo "-----------------------------------"
for dir in encut_*; do
    if [ -f "$dir/OUTCAR" ]; then
        en=$(grep "TOTEN" $dir/OUTCAR | tail -1 | awk '{print $5}')
        time=$(grep "Total CPU time" $dir/OUTCAR | awk '{print $6}')
        encut=$(echo $dir | sed 's/encut_//')
        printf "%10s | %11s | %8s\n" $encut $en $time
    fi
done

echo ""
echo "KPOINTS | Energy (eV) | Time (s)"
echo "--------------------------------"
for dir in kpoints_*; do
    if [ -f "$dir/OUTCAR" ]; then
        en=$(grep "TOTEN" $dir/OUTCAR | tail -1 | awk '{print $5}')
        time=$(grep "Total CPU time" $dir/OUTCAR | awk '{print $6}')
        k=$(echo $dir | sed 's/kpoints_//')
        printf "%7s | %11s | %8s\n" $k $en $time
    fi
done

echo ""
echo "Convergence test completed!"
echo "Check $TEST_DIR directory for details."
