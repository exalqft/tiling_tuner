#include "tuner.hpp"

int run_tuner(const StreamIndex stream_array_size) {
  printf("Reports fastest timing per kernel\n");

  const double nelem = (double)stream_array_size*
                       (double)stream_array_size*
                       (double)stream_array_size*
                       (double)stream_array_size;

  const double suN_nelem = nelem*Nc*Nc;

  const double gauge_nelem = Nd*suN_nelem;

  printf("Memory Sizes:\n");
  printf("- Gauge Array Size:  %d*%d*%" PRIu64 "^4\n",
         Nd, Nc,
         static_cast<uint64_t>(stream_array_size));
  printf("- Per complex Field:     %12.2f MB\n",
         1.0e-6 * nelem * (double)sizeof(val_t));
  printf("- Per SUNField:          %12.2f MB\n",
         1.0e-6 * suN_nelem * (double)sizeof(val_t));
  printf("- Per GaugeField:        %12.2f MB\n",
          1.0e-6 * gauge_nelem * (double)sizeof(val_t));
  printf("- Total:                 %12.2f MB\n",
         1.0e-6 * (3.0*gauge_nelem) * (double)sizeof(val_t));

  real_t total_mem = 3.0 * gauge_nelem * (double)sizeof(val_t);

  printf("Kernels will be tuned for %d iterations.\n",
          STREAM_NTIMES);

  printf(HLINE);

  printf("Initializing Views...\n");

  GaugeField a(Kokkos::view_alloc("a", Kokkos::WithoutInitializing),
                stream_array_size, stream_array_size,
                stream_array_size, stream_array_size);
  GaugeField b(Kokkos::view_alloc("b", Kokkos::WithoutInitializing),
                stream_array_size, stream_array_size,
                stream_array_size, stream_array_size);
  GaugeField c(Kokkos::view_alloc("c", Kokkos::WithoutInitializing),
                stream_array_size, stream_array_size,
                stream_array_size, stream_array_size);

  Kokkos::parallel_for(Policy<Nd>(StreamIndexArray<Nd>{0, 0, 0, 0}, 
    StreamIndexArray<Nd>{stream_array_size, stream_array_size, 
      stream_array_size, stream_array_size}),
    KOKKOS_LAMBDA(int i, int j, int k, int l) {
      #pragma unroll
      for(int mu = 0; mu < Nd; ++mu) {
        #pragma unroll
        for(int c1 = 0; c1 < Nc; ++c1) {
          #pragma unroll
          for(int c2 = 0; c2 < Nc; ++c2) {
            a(i, j, k, l, mu)[c1][c2] = val_t(1.0, 0.4);
            b(i, j, k, l, mu)[c1][c2] = val_t(2.0, 0.5);
            c(i, j, k, l, mu)[c1][c2] = val_t(3.0, 0.6);
          }
        }
      }
    });

  printf("Views Initialized.\n");
  printf(HLINE);

  int rc = 0;

  printf("Tuning...\n");

  auto best_tiling = get_tiling(StreamIndexArray<Nd>{0, 0, 0, 0}, 
    StreamIndexArray<Nd>{stream_array_size, stream_array_size, 
      stream_array_size, stream_array_size},
    KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j,
      const StreamIndex k, const StreamIndex l) {
      #pragma unroll
      for(int mu = 0; mu < Nd; ++mu) {
        a(i, j, k, l, mu) = b(i, j, k, l, mu) * c(i, j, k, l, mu);
      }
    });
  
  printf(HLINE);
  printf("Best Tile size: %d %d %d %d\n", best_tiling[0], best_tiling[1], best_tiling[2], best_tiling[3]);
  printf(HLINE);
  Kokkos::Timer timer;
  double min_time = std::numeric_limits<double>::max();
  for(int ii = 0; ii < STREAM_NTIMES; ii++) {
    timer.reset();
    Kokkos::parallel_for(Policy<4>(StreamIndexArray<Nd>{0, 0, 0, 0}, 
      StreamIndexArray<Nd>{stream_array_size, stream_array_size, 
        stream_array_size, stream_array_size}),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j,
        const StreamIndex k, const StreamIndex l) {
        #pragma unroll
        for(int mu = 0; mu < Nd; ++mu) {
          a(i, j, k, l, mu) = b(i, j, k, l, mu) * c(i, j, k, l, mu);
        }
      });
    Kokkos::fence();
    min_time = std::min(min_time, timer.seconds());
  }
  printf("rec tiling Time: %11.4e s\n", min_time);
  printf("rec tiling BW: %11.4f GB/s\n", total_mem / min_time * 1.0e-9);
  printf(HLINE);
  min_time = std::numeric_limits<double>::max();
  for(int ii = 0; ii < STREAM_NTIMES; ii++) {
    timer.reset();
    Kokkos::parallel_for(Policy<4>(StreamIndexArray<Nd>{0, 0, 0, 0}, 
      StreamIndexArray<Nd>{stream_array_size, stream_array_size, 
        stream_array_size, stream_array_size}, best_tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j,
        const StreamIndex k, const StreamIndex l) {
        #pragma unroll
        for(int mu = 0; mu < Nd; ++mu) {
          a(i, j, k, l, mu) = b(i, j, k, l, mu) * c(i, j, k, l, mu);
        }
      });
    Kokkos::fence();
    min_time = std::min(min_time, timer.seconds());
  }
  printf("tuned tiling Time: %11.4e s\n", min_time);
  printf("tuned tiling BW: %11.4f GB/s\n", total_mem / min_time * 1.0e-9);
  printf(HLINE);

  return rc;
}

int parse_args(int argc, char **argv, StreamIndex &stream_array_size) {
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
  StreamIndex stream_array_size;
  rc = parse_args(argc, argv, stream_array_size);
  if (rc == 0) {
    rc = run_tuner(stream_array_size);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}