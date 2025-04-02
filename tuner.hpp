#include<Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <getopt.h>
#include <utility>
#include <iostream>
#include <limits>

#include <sys/time.h>

#define Nc 3
#define Nd 4

#define STREAM_NTIMES 20

#define HLINE "-------------------------------------------------------------\n"

using real_t = double;
using val_t = Kokkos::complex<real_t>;
using SUN = Kokkos::Array<Kokkos::Array<val_t, Nc>, Nc>;
using GaugeField
    = Kokkos::View<SUN****[Nd], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
using Field
    = Kokkos::View<val_t****, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
template<int rank>
using Policy
    = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>;
using StreamIndex = int64_t;
template <size_t rank>
using StreamIndexArray = Kokkos::Array<StreamIndex, rank>;

KOKKOS_FORCEINLINE_FUNCTION SUN operator*(const SUN & a, const SUN & b) {
  SUN out;
  #pragma unroll
  for(int c1 = 0; c1 < Nc; ++c1){
    #pragma unroll
    for(int c2 = 0; c2 < Nc; ++c2){
      out[c1][c2] = a[c1][0] * b[0][c2];
      #pragma unroll
      for(int ci = 1; ci < Nc; ++ci){
        out[c1][c2] += a[c1][ci] * b[ci][c2];
      }
    }
  }
  return out;
}

KOKKOS_FORCEINLINE_FUNCTION SUN conj(const SUN & a) {
  SUN out;
  #pragma unroll
  for(int c1 = 0; c1 < Nc; ++c1){
    #pragma unroll
    for(int c2 = 0; c2 < Nc; ++c2){
      out[c1][c2] = Kokkos::conj(a[c2][c1]);
    }
  }
  return out;
}

template<size_t rank, class FunctorType>
auto get_tiling(const StreamIndexArray<rank> &start,
                const StreamIndexArray<rank> &end,
                const FunctorType &functor) {
  Kokkos::Timer timer;
  auto policy = Policy<rank>(start, end);
  auto tiling_rec = policy.tile_size_recommended();
  auto max_tile = policy.max_total_tile_size()/2;
  printf("Recommended Tile size: %d %d %d %d\n", tiling_rec[0], tiling_rec[1], tiling_rec[2], tiling_rec[3]);
  double time_rec = std::numeric_limits<double>::max();
  for(int ii = 0; ii < STREAM_NTIMES; ii++) {
    timer.reset();
    Kokkos::parallel_for(policy, functor);
    Kokkos::fence();
    time_rec = std::min(time_rec, timer.seconds());
  }
  printf("Time: %11.4e s\n", time_rec);
  StreamIndexArray<rank> current_tiling;
  StreamIndexArray<rank> best_tiling;
  StreamIndexArray<rank> tile_one;
  for(int i = 0; i < rank; i++) {
    current_tiling[i] = 1;
    best_tiling[i] = 1;
    tile_one[i] = 1;
  }
  double best_time = std::numeric_limits<double>::max();
  std::vector<int64_t> fast_ind_tiles;
  int64_t fast_ind = max_tile;
  while(fast_ind > 2) {
    fast_ind = fast_ind / 2;
    fast_ind_tiles.push_back(fast_ind);
  }
  for(auto &tile : fast_ind_tiles) {
    current_tiling = tile_one;
    current_tiling[0] = tile;
    int16_t second_tile = max_tile / tile;
    while(second_tile > 1) {
      current_tiling[1] = second_tile;
      if(max_tile / tile / second_tile >=4 ){
        for(int64_t i : {2, 1}) {
          current_tiling[2] = i;
          current_tiling[3] = i;
          auto tune_policy = Policy<rank>(start, end, current_tiling);
          double min_time = std::numeric_limits<double>::max();
          for(int ii = 0; ii < STREAM_NTIMES; ii++) {
            timer.reset();
            Kokkos::parallel_for(tune_policy, functor);
            Kokkos::fence();
            min_time = std::min(min_time, timer.seconds());
          }
          if(min_time < best_time) {
            best_time = min_time;
            best_tiling = current_tiling;
          }
          printf("Current Tile size: %d %d %d %d, time: %11.4e\n", current_tiling[0], current_tiling[1], current_tiling[2], current_tiling[3], min_time);
        }
      }else if(max_tile / tile / second_tile == 2) {
        for(int64_t i : {2, 1}) {
          current_tiling[2] = i;
          current_tiling[3] = 1;
          auto tune_policy = Policy<rank>(start, end, current_tiling);
          double min_time = std::numeric_limits<double>::max();
          for(int ii = 0; ii < STREAM_NTIMES; ii++) {
            timer.reset();
            Kokkos::parallel_for(tune_policy, functor);
            Kokkos::fence();
            min_time = std::min(min_time, timer.seconds());
          }
          if(min_time < best_time) {
            best_time = min_time;
            best_tiling = current_tiling;
          }
          printf("Current Tile size: %d %d %d %d, time: %11.4e\n", current_tiling[0], current_tiling[1], current_tiling[2], current_tiling[3], min_time);
        }
      }else {
        current_tiling[2] = 1;
        current_tiling[3] = 1;
        auto tune_policy = Policy<rank>(start, end, current_tiling);
        double min_time = std::numeric_limits<double>::max();
        for(int ii = 0; ii < STREAM_NTIMES; ii++) {
          timer.reset();
          Kokkos::parallel_for(tune_policy, functor);
          Kokkos::fence();
          min_time = std::min(min_time, timer.seconds());
        }
        if(min_time < best_time) {
          best_time = min_time;
          best_tiling = current_tiling;
        }
        printf("Current Tile size: %d %d %d %d, time:%11.4e\n", current_tiling[0], current_tiling[1], current_tiling[2], current_tiling[3], min_time);
      }
      second_tile = second_tile / 2;
    }
  }
  printf("Final results:\n");
  printf("Best Tile size: %d %d %d %d\n", best_tiling[0], best_tiling[1], best_tiling[2], best_tiling[3]);
  printf("Best Time: %11.4e s\n", best_time);
  printf("Time with recommended tile size: %11.4e s\n", time_rec);
  printf("Speedup: %f\n", time_rec / best_time);
  return best_tiling;
};