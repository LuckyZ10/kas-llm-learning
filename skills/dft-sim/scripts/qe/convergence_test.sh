#!/bin/bash
# QE收敛性测试脚本

PSEUDO_DIR="./pseudo"
TEST_DIR="qe_convergence_test"
PREFIX="test"

mkdir -p $TEST_DIR
cd $TEST_DIR

# 1. 截断能测试
echo "=== ecutwfc Convergence Test ==="
for ECUT in 20 30 40 50 60 80; do
    mkdir -p ecut_$ECUT
    cd ecut_$ECUT
    
    cat > scf.in << EOF
&CONTROL
  calculation = 'scf'
  prefix = '$PREFIX'
  outdir = './tmp'
  pseudo_dir = '$PSEUDO_DIR'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = $ECUT
  ecutrho = $(($ECUT * 8))
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0D-8
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
6 6 6 0 0 0
EOF
    
    echo "Testing ecutwfc = $ECUT Ry"
    # pw.x -in scf.in > scf.out 2>&1
    
    cd ..
done

# 2. k点网格测试
echo ""
echo "=== k-points Convergence Test ==="
for K in 2 4 6 8 10 12; do
    mkdir -p k_${K}x${K}x${K}
    cd k_${K}x${K}x${K}
    
    cat > scf.in << EOF
&CONTROL
  calculation = 'scf'
  prefix = '$PREFIX'
  outdir = './tmp'
  pseudo_dir = '$PSEUDO_DIR'
/
&SYSTEM
  ibrav = 2
  celldm(1) = 10.26
  nat = 2
  ntyp = 1
  ecutwfc = 40
  ecutrho = 320
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.01
/
&ELECTRONS
  conv_thr = 1.0D-8
/
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
$K $K $K 0 0 0
EOF
    
    echo "Testing k-points = ${K}x${K}x${K}"
    # pw.x -in scf.in > scf.out 2>&1
    
    cd ..
done

# 3. 提取结果
echo ""
echo "=== Results Summary ==="
echo "ecutwfc (Ry) | Energy (Ry) | Time (s)"
echo "-------------------------------------"
for dir in ecut_*; do
    if [ -f "$dir/scf.out" ]; then
        en=$(grep "!" $dir/scf.out | tail -1 | awk '{print $5}')
        time=$(grep "PWSCF" $dir/scf.out | tail -1 | awk '{print $4}')
        ecut=$(echo $dir | sed 's/ecut_//')
        printf "%12s | %11s | %8s\n" $ecut $en $time
    fi
done

echo ""
echo "k-points | Energy (Ry) | Time (s)"
echo "---------------------------------"
for dir in k_*x*x*; do
    if [ -f "$dir/scf.out" ]; then
        en=$(grep "!" $dir/scf.out | tail -1 | awk '{print $5}')
        time=$(grep "PWSCF" $dir/scf.out | tail -1 | awk '{print $4}')
        k=$(echo $dir | sed 's/k_//' | sed 's/x/ /g')
        printf "%8s | %11s | %8s\n" "$k" $en $time
    fi
done
