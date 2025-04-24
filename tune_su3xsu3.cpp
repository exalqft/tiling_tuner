#include "tuner.hpp"
#include <getopt.h>

#define Ndims 4
#define Ncolor 3

// define related fields
template <size_t Nc>
using SUN = Kokkos::Array<Kokkos::Array<complex_t,Nc>,Nc>;
template <size_t Nd, size_t Nc>
using GaugeField = Kokkos::View<SUN<Nc>****[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;
// define corresponding constant fields
#if defined ( KOKKOS_ENABLE_CUDA )
template <size_t Nd, size_t Nc>
using constGaugeField = Kokkos::View<const SUN<Nc>****[Nd], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
#else
template <size_t Nd, size_t Nc>
using constGaugeField = Kokkos::View<const SUN<Nc>****[Nd], Kokkos::MemoryTraits<Kokkos::Restrict>>;
#endif

// su3 matrix multiplication
template <size_t Nc>
KOKKOS_FORCEINLINE_FUNCTION
SUN<Nc> operator*(const SUN<Nc> &a, const SUN<Nc> &b) {
  SUN<Nc> c;
  #pragma unroll
  for (size_t i = 0; i < Nc; ++i) {
    #pragma unroll
    for (size_t j = 0; j < Nc; ++j) {
      c[i][j] = a[i][0] * b[0][j];
      #pragma unroll
      for (size_t k = 1; k < Nc; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

// functor for c = a*b
template <size_t Nd, size_t Nc>
struct su3xsu3 {
  constGaugeField<Nd, Nc> a;
  constGaugeField<Nd, Nc> b;
  GaugeField<Nd, Nc> c;

  su3xsu3(GaugeField<Nd, Nc> a_, constGaugeField<Nd, Nc> b_,
          GaugeField<Nd, Nc> c_) : a(a_), b(b_), c(c_) {}

  KOKKOS_FORCEINLINE_FUNCTION
  void operator()(const index_t i0, const index_t i1,
                  const index_t i2, const index_t i3) const {
    #pragma unroll
    for (index_t mu = 0; mu < Nd; ++mu) {
      c(i0, i1, i2, i3, mu) = a(i0, i1, i2, i3, mu) * b(i0, i1, i2, i3, mu);
    }
  }
};

template <size_t Nd, size_t Nc>
int run_tuner(const index_t stream_array_size) {
  printf("Reports fastest timing per kernel\n");

  const real_t nelem = (real_t)stream_array_size*
                       (real_t)stream_array_size*
                       (real_t)stream_array_size*
                       (real_t)stream_array_size;

  const real_t suN_nelem = nelem*Nc*Nc;

  const real_t gauge_nelem = Nd*suN_nelem;

  printf("Memory Sizes:\n");
  printf("- Gauge Array Size:  %d*%d*%" PRIu64 "^4\n",
         Nd, Nc,
         static_cast<uint64_t>(stream_array_size));
  printf("- Per complex Field:     %12.2f MB\n",
         1.0e-6 * nelem * (real_t)sizeof(complex_t));
  printf("- Per SUNField:          %12.2f MB\n",
         1.0e-6 * suN_nelem * (real_t)sizeof(complex_t));
  printf("- Per GaugeField:        %12.2f MB\n",
          1.0e-6 * gauge_nelem * (real_t)sizeof(complex_t));
  printf("- Total:                 %12.2f MB\n",
         1.0e-6 * (3.0*gauge_nelem) * (real_t)sizeof(complex_t));

  real_t total_mem = 3.0 * gauge_nelem * (real_t)sizeof(complex_t);

  printf("Kernels will be tuned for %d iterations.\n",
          STREAM_NTIMES);

  printf(HLINE);

  printf("Initializing Views...\n");

  GaugeField<Nd,Nc> a(Kokkos::view_alloc("a", Kokkos::WithoutInitializing),
                             stream_array_size, stream_array_size,
                             stream_array_size, stream_array_size);
  GaugeField<Nd,Nc> b(Kokkos::view_alloc("b", Kokkos::WithoutInitializing),
                             stream_array_size, stream_array_size,
                             stream_array_size, stream_array_size);
  GaugeField<Nd,Nc> c(Kokkos::view_alloc("c", Kokkos::WithoutInitializing),
                             stream_array_size, stream_array_size,
                             stream_array_size, stream_array_size);

  Kokkos::parallel_for(Policy<Nd>(IndexArray<Nd>{0, 0, 0, 0}, 
    IndexArray<Nd>{stream_array_size, stream_array_size, 
      stream_array_size, stream_array_size}),
    KOKKOS_LAMBDA(index_t i, index_t j, index_t k, index_t l) {
      #pragma unroll
      for(index_t mu = 0; mu < Nd; ++mu) {
        #pragma unroll
        for(index_t c1 = 0; c1 < Nc; ++c1) {
          #pragma unroll
          for(index_t c2 = 0; c2 < Nc; ++c2) {
            a(i, j, k, l, mu)[c1][c2] = complex_t(1.0, 0.4);
            b(i, j, k, l, mu)[c1][c2] = complex_t(2.0, 0.5);
            c(i, j, k, l, mu)[c1][c2] = complex_t(3.0, 0.6);
          }
        }
      }
    });

  printf("Views Initialized.\n");
  printf(HLINE);

  int rc = 0;

 su3xsu3<Nd, Nc> su3mult(a, b, c);

  printf("Tuning...\n");

  auto best_tiling = tune_parallel_for<Nd>(
    IndexArray<Nd>{0, 0, 0, 0}, 
    IndexArray<Nd>{stream_array_size, stream_array_size, 
      stream_array_size, stream_array_size},
    su3mult);

  printf(HLINE);
  printf("Best Tile size: %d %d %d %d\n", best_tiling[0], best_tiling[1], best_tiling[2], best_tiling[3]);
  printf(HLINE);
  Kokkos::Timer timer;
  real_t min_time = std::numeric_limits<real_t>::max();
  for(int ii = 0; ii < STREAM_NTIMES; ii++) {
    timer.reset();
    Kokkos::parallel_for(Policy<4>(IndexArray<Nd>{0, 0, 0, 0}, 
      IndexArray<Nd>{stream_array_size, stream_array_size, 
        stream_array_size, stream_array_size}),
      su3mult);
    Kokkos::fence();
    min_time = std::min(min_time, timer.seconds());
  }
  printf("rec tiling Time: %11.4e s\n", min_time);
  printf("rec tiling BW: %11.4f GB/s\n", total_mem / min_time * 1.0e-9);
  printf(HLINE);
  min_time = std::numeric_limits<real_t>::max();
  for(int ii = 0; ii < STREAM_NTIMES; ii++) {
    timer.reset();
    Kokkos::parallel_for(Policy<4>(IndexArray<Nd>{0, 0, 0, 0}, 
      IndexArray<Nd>{stream_array_size, stream_array_size, 
        stream_array_size, stream_array_size}, best_tiling),
      su3mult);
    Kokkos::fence();
    min_time = std::min(min_time, timer.seconds());
  }
  printf("tuned tiling Time: %11.4e s\n", min_time);
  printf("tuned tiling BW: %11.4f GB/s\n", total_mem / min_time * 1.0e-9);
  printf(HLINE);

  return rc;
}

int parse_args(int argc, char **argv, index_t &stream_array_size) {
  // Defaults
  stream_array_size = 32;

  const std::string help_string =
      "  -n <N>, --nelements <N>\n"
      "     Create stream views containing [4][Nc][Nc]<N>^4 elements.\n"
      "     Default: 32\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"nelements", required_argument, NULL, 'n'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "n:h", long_options, &option_index)) !=
         -1)
    switch (c) {
      case 'n': stream_array_size = atoi(optarg); break;
      case 'h':
        printf("%s", help_string.c_str());
        return -2;
        break;
      case 0: break;
      default:
        printf("%s", help_string.c_str());
        return -1;
        break;
    }
  return 0;
}

int main(int argc, char *argv[]) {
  printf(HLINE);
  printf("Kokkos 4D GaugeField (mu static, SUN as Kokkos::Array) MDRangePolicy tiling tuner Benchmark\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  int rc;
  index_t stream_array_size;
  rc = parse_args(argc, argv, stream_array_size);
  if (rc == 0) {
    rc = run_tuner<Ndims,Ncolor>(stream_array_size);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}