#!/bin/bash
# QE批量计算脚本

# 配置
MAX_JOBS=4
PW_EXEC="pw.x"
PH_EXEC="ph.x"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数: 运行PW计算
run_pw() {
    local input=$1
    local dir=$(dirname "$input")
    local base=$(basename "$input" .in)
    
    cd "$dir" || return 1
    
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] Running: $input${NC}"
    
    mpirun -np 4 "$PW_EXEC" -in "$base.in" > "$base.out" 2>&1
    
    if grep -q "convergence has been achieved" "$base.out"; then
        echo -e "${GREEN}  ✓ Converged${NC}"
        return 0
    else
        echo -e "${RED}  ✗ Failed${NC}"
        return 1
    fi
    
    cd - > /dev/null
}

# 函数: 运行PH计算
run_ph() {
    local input=$1
    local dir=$(dirname "$input")
    local base=$(basename "$input" .in)
    
    cd "$dir" || return 1
    
    echo -e "${YELLOW}[$(date '+%H:%M:%S')] Running phonon: $input${NC}"
    
    mpirun -np 4 "$PH_EXEC" -in "$base.in" > "$base.out" 2>&1
    
    if grep -q "Finished" "$base.out"; then
        echo -e "${GREEN}  ✓ Completed${NC}"
        return 0
    else
        echo -e "${RED}  ✗ Failed${NC}"
        return 1
    fi
    
    cd - > /dev/null
}

# 主程序
echo "======================================"
echo "QE Batch Calculation Script"
echo "Started: $(date)"
echo "======================================"

# 查找所有输入文件
PW_INPUTS=($(find . -name "pw.in" -o -name "scf.in" | sort))
PH_INPUTS=($(find . -name "ph.in" | sort))

echo "Found ${#PW_INPUTS[@]} PW calculations"
echo "Found ${#PH_INPUTS[@]} PH calculations"
echo ""

# 运行PW计算
echo "--- PWscf Calculations ---"
for input in "${PW_INPUTS[@]}"; do
    run_pw "$input"
done

echo ""
echo "--- Phonon Calculations ---"
for input in "${PH_INPUTS[@]}"; do
    run_ph "$input"
done

echo ""
echo "======================================"
echo "All calculations completed!"
echo "Finished: $(date)"
echo "======================================"
