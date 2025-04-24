# size of 4D lattices
n_vals=(4 6 8 10 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68)

source ${SRCDIR}/compilation/amd_mi250/load_modules.sh

echo nt n tiling bw > ${SRCDIR}/run_scripts_and_results/amd_mi250/results.dat

export ROCR_VISIBLE_DEVICES=0

# 8 GCDs per node, one 64 core CPU
# 1 core reserved, 1 GCD -> 7 cores
for nt in 7;
do

  export OMP_NUM_THREADS=${nt}
  export OMP_PLACES=cores
  export OMP_PROC_BIND=close

  for n in ${n_vals[@]};
  do

    results=( $(${BUILDDIR}/amd_mi250/tune_su3xsu3 -n ${n} | grep GB/s | awk '{print $1 " " $4}') )
    for i in $(seq 0 5);
    do
      echo ${nt} ${n} ${results[2*i]} ${results[2*i+1]} >> ${SRCDIR}/run_scripts_and_results/amd_mi250/results.dat
    done

  done
done

    
