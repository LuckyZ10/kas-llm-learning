#!/bin/bash
# LAMMPS自动化工作流脚本
# workflow_manager.sh

# 工作流配置
WORKFLOW_DIR=$(pwd)
RESULTS_DIR="${WORKFLOW_DIR}/results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$RESULTS_DIR/workflow.log"
}

# 错误处理
error_exit() {
    log "ERROR: $1"
    exit 1
}

# ============================================
# 阶段1: 系统准备
# ============================================
stage1_preparation() {
    log "=== Stage 1: System Preparation ==="
    
    local system_type=$1
    local output_prefix="$RESULTS_DIR/stage1"
    
    case $system_type in
        metal)
            log "Creating metal system (Cu)"
            python3 "${WORKFLOW_DIR}/scripts/generate_metal.py" \
                --element Cu --lattice fcc --size 10 10 10 \
                --output "${output_prefix}.data" || error_exit "Metal generation failed"
            ;;
        polymer)
            log "Creating polymer system (PE)"
            python3 "${WORKFLOW_DIR}/scripts/generate_polymer.py" \
                --monomers 100 --chains 10 \
                --output "${output_prefix}.data" || error_exit "Polymer generation failed"
            ;;
        water)
            log "Creating water system"
            python3 "${WORKFLOW_DIR}/scripts/generate_water.py" \
                --density 1.0 --nmol 1000 \
                --output "${output_prefix}.data" || error_exit "Water generation failed"
            ;;
        interface)
            log "Creating interface system"
            python3 "${WORKFLOW_DIR}/scripts/generate_interface.py" \
                --substrate Cu --adsorbate H2O \
                --output "${output_prefix}.data" || error_exit "Interface generation failed"
            ;;
        *)
            error_exit "Unknown system type: $system_type"
            ;;
    esac
    
    log "Stage 1 completed: ${output_prefix}.data"
}

# ============================================
# 阶段2: 能量最小化
# ============================================
stage2_minimization() {
    log "=== Stage 2: Energy Minimization ==="
    
    local input_data=$1
    local output_prefix="$RESULTS_DIR/stage2"
    
    # 生成最小化输入文件
    cat > "${output_prefix}.in" << EOF
# Energy Minimization
units metal
atom_style atomic
boundary p p p

read_data ${input_data}

# 势函数 (根据体系自动选择)
include ${WORKFLOW_DIR}/potentials/auto_select.inc

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# 输出
dump 1 all custom 1000 ${output_prefix}.dump id type x y z
dump_modify 1 sort id

thermo 100
thermo_style custom step pe press vol

# 最小化
minimize 1.0e-12 1.0e-12 1000 100000

write_data ${output_prefix}.data
write_restart ${output_prefix}.restart
EOF

    # 运行最小化
    mpirun -np ${NP:-4} lmp -in "${output_prefix}.in" -log "${output_prefix}.log"
    
    # 检查收敛
    if grep -q "Energy convergence" "${output_prefix}.log"; then
        log "Stage 2 completed successfully"
    else
        log "WARNING: Minimization may not have fully converged"
    fi
}

# ============================================
# 阶段3: 热平衡 (NVT/NPT)
# ============================================
stage3_equilibration() {
    log "=== Stage 3: Thermal Equilibration ==="
    
    local input_data=$1
    local ensemble=${2:-npt}
    local temperature=${3:-300}
    local pressure=${4:-1.0}
    local output_prefix="$RESULTS_DIR/stage3_${ensemble}"
    
    # 生成平衡输入文件
    cat > "${output_prefix}.in" << EOF
# Thermal Equilibration
units metal
atom_style atomic
boundary p p p

read_data ${input_data}

include ${WORKFLOW_DIR}/potentials/auto_select.inc

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# 初始速度
velocity all create ${temperature}.0 12345 mom yes rot yes dist gaussian

# 输出
dump 1 all custom 1000 ${output_prefix}.dump id type x y z vx vy vz
dump_modify 1 sort id

thermo 1000
thermo_style custom step temp pe ke etotal press vol density

# 系综选择
EOF

    if [ "$ensemble" == "nvt" ]; then
        cat >> "${output_prefix}.in" << EOF
fix 1 all nvt temp ${temperature}.0 ${temperature}.0 \$(100.0*dt)
EOF
    else
        cat >> "${output_prefix}.in" << EOF
fix 1 all npt temp ${temperature}.0 ${temperature}.0 \$(100.0*dt) iso ${pressure}.0 ${pressure}.0 \$(1000.0*dt)
EOF
    fi

    cat >> "${output_prefix}.in" << EOF

# 运行
timestep 0.001
run 50000

write_data ${output_prefix}.data
write_restart ${output_prefix}.restart
EOF

    # 运行平衡
    mpirun -np ${NP:-4} lmp -in "${output_prefix}.in" -log "${output_prefix}.log"
    
    log "Stage 3 completed: ${ensemble} at ${temperature}K"
}

# ============================================
# 阶段4: 生产运行
# ============================================
stage4_production() {
    log "=== Stage 4: Production Run ==="
    
    local input_data=$1
    local ensemble=$2
    local temperature=$3
    local steps=${4:-100000}
    local output_prefix="$RESULTS_DIR/stage4_prod"
    
    cat > "${output_prefix}.in" << EOF
# Production Run
units metal
atom_style atomic
boundary p p p

read_restart ${input_data}

include ${WORKFLOW_DIR}/potentials/auto_select.inc

neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# 输出设置 - 更频繁
thermo 1000
thermo_style custom step temp pe ke etotal press vol density cpu

# 轨迹输出
dump 1 all custom 5000 ${output_prefix}.dump id type x y z vx vy vz fx fy fz
dump_modify 1 sort id

# 计算量
coordinate_rdf all rdf 200 1 1
coordinate_msd all msd
coordinate_pe all pe/atom

fix 2 all ave/time 100 10 1000 c_rdf[*] file ${output_prefix}.rdf mode vector
fix 3 all ave/time 10 100 1000 c_msd[*] file ${output_prefix}.msd mode vector

# 系综
fix 1 all ${ensemble} temp ${temperature}.0 ${temperature}.0 \$(100.0*dt)

# 运行
timestep 0.001
run ${steps}

write_restart ${output_prefix}.restart
write_data ${output_prefix}.data
EOF

    # 运行生产
    mpirun -np ${NP:-4} lmp -in "${output_prefix}.in" -log "${output_prefix}.log"
    
    log "Stage 4 completed: ${steps} steps"
}

# ============================================
# 阶段5: 分析
# ============================================
stage5_analysis() {
    log "=== Stage 5: Analysis ==="
    
    local traj_file=$1
    local output_prefix="$RESULTS_DIR/stage5"
    
    # RDF分析
    python3 "${WORKFLOW_DIR}/scripts/analyze_rdf.py" \
        --input "${output_prefix}.rdf" \
        --output "${output_prefix}_rdf.png" || log "RDF analysis failed"
    
    # MSD分析 - 扩散系数
    python3 "${WORKFLOW_DIR}/scripts/analyze_diffusion.py" \
        --input "${output_prefix}.msd" \
        --output "${output_prefix}_diffusion.png" || log "MSD analysis failed"
    
    # 结构分析
    python3 "${WORKFLOW_DIR}/scripts/analyze_structure.py" \
        --trajectory "$traj_file" \
        --output "${output_prefix}_structure" || log "Structure analysis failed"
    
    # 生成报告
    python3 "${WORKFLOW_DIR}/scripts/generate_report.py" \
        --results "$RESULTS_DIR" \
        --output "${RESULTS_DIR}/final_report.html"
    
    log "Stage 5 completed"
}

# ============================================
# 主函数
# ============================================
main() {
    log "Starting LAMMPS Workflow Manager"
    log "Results directory: $RESULTS_DIR"
    
    # 参数
    SYSTEM_TYPE=${1:-metal}
    ENSEMBLE=${2:-npt}
    TEMPERATURE=${3:-300}
    STEPS=${4:-100000}
    
    log "Parameters: system=$SYSTEM_TYPE, ensemble=$ENSEMBLE, T=$TEMPERATURE, steps=$STEPS"
    
    # 执行各阶段
    stage1_preparation "$SYSTEM_TYPE"
    stage2_minimization "$RESULTS_DIR/stage1.data"
    stage3_equilibration "$RESULTS_DIR/stage2.data" "$ENSEMBLE" "$TEMPERATURE"
    stage4_production "$RESULTS_DIR/stage3_${ENSEMBLE}.restart" "$ENSEMBLE" "$TEMPERATURE" "$STEPS"
    stage5_analysis "$RESULTS_DIR/stage4_prod.dump"
    
    log "=== Workflow completed successfully ==="
    log "Results available at: $RESULTS_DIR"
}

# 运行
main "$@"
