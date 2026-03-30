#!/bin/bash
# 批量VASP计算提交脚本
# 遍历所有子目录并提交计算

# 配置
MAX_JOBS=5          # 最大同时运行作业数
SLEEP_TIME=30       # 检查间隔(秒)
VASP_EXEC="vasp_std"

# 函数: 检查运行中的作业数
count_running_jobs() {
    ps aux | grep "$VASP_EXEC" | grep -v grep | wc -l
}

# 函数: 运行计算
run_calculation() {
    local dir=$1
    cd "$dir" || return 1
    
    echo "[$(date '+%H:%M:%S')] Starting: $dir"
    
    # 检查必要文件
    if [[ ! -f POSCAR ]] || [[ ! -f INCAR ]] || [[ ! -f KPOINTS ]]; then
        echo "  Error: Missing input files in $dir"
        cd - > /dev/null
        return 1
    fi
    
    # 运行VASP
    mpirun -np 4 "$VASP_EXEC" > vasp.out 2>&1
    
    # 检查收敛
    if grep -q "reached required accuracy" OUTCAR; then
        echo "  ✓ Converged"
        echo "$(date) - $dir - Converged" >> ../batch_run.log
    else
        echo "  ✗ Not converged"
        echo "$(date) - $dir - Failed" >> ../batch_run.log
    fi
    
    cd - > /dev/null
}

# 主程序
echo "======================================"
echo "VASP Batch Calculation Script"
echo "Started: $(date)"
echo "======================================"

# 获取所有计算目录
DIRS=($(find . -maxdepth 2 -type d -name "*calc*" -o -name "Si_*" | sort))
TOTAL=${#DIRS[@]}

echo "Found $TOTAL calculation directories"
echo ""

# 处理每个目录
for i in "${!DIRS[@]}"; do
    dir="${DIRS[$i]}"
    
    # 检查是否已计算
    if [[ -f "$dir/OUTCAR" ]]; then
        echo "[$((i+1))/$TOTAL] $dir - Already done, skipping"
        continue
    fi
    
    # 等待直到有可用槽位
    while true; do
        RUNNING=$(count_running_jobs)
        if [[ $RUNNING -lt $MAX_JOBS ]]; then
            break
        fi
        echo "  Waiting... ($RUNNING jobs running)"
        sleep $SLEEP_TIME
    done
    
    # 提交计算 (后台运行)
    echo "[$((i+1))/$TOTAL] Submitting: $dir"
    run_calculation "$dir" &
    
    sleep 2
done

# 等待所有后台作业完成
wait

echo ""
echo "======================================"
echo "All calculations completed!"
echo "Finished: $(date)"
echo "Log: batch_run.log"
echo "======================================"
