# size of 4D lattices
n_vals=(4 6 8 10 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68)

source ${SRCDIR}/compilation/dual_epyc2_7742_gcc/load_modules.sh

echo nt n tiling bw > ${SRCDIR}/run_scripts_and_results/dual_epyc2_7742_gcc/results.dat

# 64 threads, close binding -> use a single socket
# 128 threads -> both sockets
for nt in 64 128;
do

  export OMP_NUM_THREADS=${nt}
  export OMP_PLACES=cores
  export OMP_PROC_BIND=close

  for n in ${n_vals[@]};
  do

    results=( $(${BUILDDIR}/dual_epyc2_7742_gcc/tune_su3xsu3 -n ${n} | grep GB/s | awk '{print $1 " " $4}') )
    for i in $(seq 0 5);
    do
      echo ${nt} ${n} ${results[2*i]} ${results[2*i+1]} >> ${SRCDIR}/run_scripts_and_results/dual_epyc2_7742_gcc/results.dat
    done
    
  done
done

    
