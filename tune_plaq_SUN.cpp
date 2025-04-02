#include "tuner.hpp"

using SUNField = Kokkos::View<SUN****, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
using GaugeMu = Kokkos::Array<SUNField, Nd>;

KOKKOS_FORCEINLINE_FUNCTION SUN operator+(const SUN & a, const SUN & b) {
  SUN out;
  #pragma unroll
  for(int c1 = 0; c1 < Nc; ++c1){
    #pragma unroll
    for(int c2 = 0; c2 < Nc; ++c2){
      out[c1][c2] = a[c1][c2] + b[c1][c2];
    }
  }
  return out;
}

KOKKOS_FORCEINLINE_FUNCTION 
val_t 
plaq_site(const SUN &a, const SUN &b,
          const SUN &c, const SUN &d) {
  val_t tmu = 0;
  #pragma unroll
  for(int c_out = 0; c_out < Nc; ++c_out) {
    #pragma unroll
    for(int c_in_1 = 0; c_in_1 < Nc; ++c_in_1) {
      #pragma unroll
      for(int c_in_2 = 0; c_in_2 < Nc; ++c_in_2) {
        #pragma unroll
        for(int c_in_3 = 0; c_in_3 < Nc; ++c_in_3) {
          tmu += a[c_out][c_in_1] * b[c_in_1][c_in_2] * 
                 Kokkos::conj(c[c_out][c_in_3] * d[c_in_3][c_in_2]);
        }
      }
    } 
  }
  return tmu;
}

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
         1.0e-6 * (nelem+gauge_nelem) * (double)sizeof(val_t));

  real_t total_mem = (gauge_nelem + nelem) * (double)sizeof(val_t);

  printf("Kernels will be tuned for %d iterations.\n",
          STREAM_NTIMES);

  printf(HLINE);

  printf("Initializing Views...\n");

  GaugeMu g;
  for(int ii = 0; ii < Nd; ++ii) {
    g[ii] = SUNField(Kokkos::view_alloc("g", Kokkos::WithoutInitializing),
                       stream_array_size, stream_array_size,
                       stream_array_size, stream_array_size);
  }
  Field temp(Kokkos::view_alloc("temp", Kokkos::WithoutInitializing),
                       stream_array_size, stream_array_size,
                       stream_array_size, stream_array_size);

  Kokkos::parallel_for(Policy<Nd>(StreamIndexArray<Nd>{0, 0, 0, 0}, 
    StreamIndexArray<Nd>{stream_array_size, stream_array_size, 
      stream_array_size, stream_array_size}),
    KOKKOS_LAMBDA(int i, int j, int k, int l) {
      #pragma unroll
      for(int ii = 0; ii < Nd; ++ii) {
        #pragma unroll
        for(int c1 = 0; c1 < Nc; ++c1) {
          #pragma unroll
          for(int c2 = 0; c2 < Nc; ++c2) {
            g[ii](i, j, k, l)[c1][c2] = val_t(1.0, 0.4);
            temp(i, j, k, l) = val_t(0.0, 0.0);
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
      const StreamIndex ip = (i + 1) % stream_array_size;
      const StreamIndex jp = (j + 1) % stream_array_size;
      const StreamIndex kp = (k + 1) % stream_array_size;
      const StreamIndex lp = (l + 1) % stream_array_size;
      temp(i, j, k, l) = plaq_site(g[0](i, j, k, l), g[1](ip, j, k, l), g[1](i, j, k, l), g[0](i, jp, k, l));
                        + plaq_site(g[0](i, j, k, l), g[2](ip, j, k, l), g[2](i, j, k, l), g[0](i, j, kp, l))
                        + plaq_site(g[0](i, j, k, l), g[3](ip, j, k, l), g[3](i, j, k, l), g[0](i, j, lp, l))
                        + plaq_site(g[1](i, j, k, l), g[2](i, jp, k, l), g[2](i, j, k, l), g[1](i, j, kp, l))
                        + plaq_site(g[1](i, j, k, l), g[3](i, jp, k, l), g[3](i, j, k, l), g[1](i, j, k, lp))
                        + plaq_site(g[2](i, j, k, l), g[3](i, j, kp, l), g[3](i, j, k, l), g[2](i, j, k, lp));
    });
  
  printf(HLINE);
  printf("Best Tile size: %d %d %d %d\n", best_tiling[0], best_tiling[1], best_tiling[2], best_tiling[3]);
  printf(HLINE);
  Kokkos::Timer timer;
  double min_time = std::numeric_limits<double>::max();
  for(int ii = 0; ii < STREAM_NTIMES; ii++) {
    timer.reset();
    Kokkos::parallel_for(Policy<Nd>(StreamIndexArray<Nd>{0, 0, 0, 0}, 
      StreamIndexArray<Nd>{stream_array_size, stream_array_size, 
        stream_array_size, stream_array_size}),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j,
        const StreamIndex k, const StreamIndex l) {
        const StreamIndex ip = (i + 1) % stream_array_size;
        const StreamIndex jp = (j + 1) % stream_array_size;
        const StreamIndex kp = (k + 1) % stream_array_size;
        const StreamIndex lp = (l + 1) % stream_array_size;
        temp(i, j, k, l) = plaq_site(g[0](i, j, k, l), g[1](ip, j, k, l), g[1](i, j, k, l), g[0](i, jp, k, l));
                          + plaq_site(g[0](i, j, k, l), g[2](ip, j, k, l), g[2](i, j, k, l), g[0](i, j, kp, l))
                          + plaq_site(g[0](i, j, k, l), g[3](ip, j, k, l), g[3](i, j, k, l), g[0](i, j, lp, l))
                          + plaq_site(g[1](i, j, k, l), g[2](i, jp, k, l), g[2](i, j, k, l), g[1](i, j, kp, l))
                          + plaq_site(g[1](i, j, k, l), g[3](i, jp, k, l), g[3](i, j, k, l), g[1](i, j, k, lp))
                          + plaq_site(g[2](i, j, k, l), g[3](i, j, kp, l), g[3](i, j, k, l), g[2](i, j, k, lp));
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
    Kokkos::parallel_for(Policy<Nd>(StreamIndexArray<Nd>{0, 0, 0, 0}, 
      StreamIndexArray<Nd>{stream_array_size, stream_array_size, 
        stream_array_size, stream_array_size}, best_tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j,
        const StreamIndex k, const StreamIndex l) {
        const StreamIndex ip = (i + 1) % stream_array_size;
        const StreamIndex jp = (j + 1) % stream_array_size;
        const StreamIndex kp = (k + 1) % stream_array_size;
        const StreamIndex lp = (l + 1) % stream_array_size;
        temp(i, j, k, l) = plaq_site(g[0](i, j, k, l), g[1](ip, j, k, l), g[1](i, j, k, l), g[0](i, jp, k, l));
                         + plaq_site(g[0](i, j, k, l), g[2](ip, j, k, l), g[2](i, j, k, l), g[0](i, j, kp, l))
                         + plaq_site(g[0](i, j, k, l), g[3](ip, j, k, l), g[3](i, j, k, l), g[0](i, j, lp, l))
                         + plaq_site(g[1](i, j, k, l), g[2](i, jp, k, l), g[2](i, j, k, l), g[1](i, j, kp, l))
                         + plaq_site(g[1](i, j, k, l), g[3](i, jp, k, l), g[3](i, j, k, l), g[1](i, j, k, lp))
                         + plaq_site(g[2](i, j, k, l), g[3](i, j, kp, l), g[3](i, j, k, l), g[2](i, j, k, lp));
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
  printf("Kokkos GaugeField as 4D vector of SUNField with SUN as Kokkos array. MDRangePolicy tiling tuner Benchmark\n");
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