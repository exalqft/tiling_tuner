source ${SRCDIR}/compilation/dual_epyc2_7742_gcc/load_modules.sh

cmake \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DKokkos_ARCH_ZEN2=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  ${SRCDIR}
