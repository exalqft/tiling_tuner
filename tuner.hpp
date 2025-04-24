#pragma once
#include<Kokkos_Core.hpp>

#define HLINE "-------------------------------------------------------------\n"
#define STREAM_NTIMES 20

using real_t = double;
using complex_t = Kokkos::complex<real_t>;
using index_t = int;
template <size_t rank>
using IndexArray = Kokkos::Array<index_t, rank>;
template <size_t rank>
using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>;

template<size_t rank, class FunctorType>
IndexArray<rank> tune_parallel_for(const IndexArray<rank> &start,
                         const IndexArray<rank> &end,
                         const FunctorType &functor) {
  // define the policy
  const auto policy = Policy<rank>(start, end);
  // array to store the best tiling
  // initialise to 1
  IndexArray<rank> best_tiling;
  for(index_t i = 0; i < rank; i++) {
    best_tiling[i] = 1;
  }
  // timer for tuning
  Kokkos::Timer timer;
  double best_time = std::numeric_limits<double>::max();
  // first for hostspace
  // there is no tuning
  if constexpr (std::is_same_v<typename Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>) {
    // for OpenMP we parallelise over the two outermost (leftmost) dimensions and so the chunk size
    // for the innermost dimensions corresponds to the view extents
    best_tiling[rank-1] = end[rank-1] - start[rank-1];
    best_tiling[rank-2] = end[rank-2] - start[rank-2];
  }else {
    // for Cuda we need to tune the tiling
    // first get the max allowed tile size
    // this is divided by 2, since from testing I found that
    // when the product of the tiling matches the max tile size
    // the timing makes no sense, not sure why
    // need to test further
    const auto max_tile = policy.max_total_tile_size()/2;
    IndexArray<rank> current_tiling;
    IndexArray<rank> tile_one;
    for(index_t i = 0; i < rank; i++) {
      current_tiling[i] = 1;
      best_tiling[i] = 1;
      tile_one[i] = 1;
    }
    // a vector to store the tiling sizes
    // for the fastest running index
    std::vector<index_t> fast_ind_tiles;
    index_t fast_ind = max_tile;
    while(fast_ind > 2) {
      fast_ind = fast_ind / 2;
      fast_ind_tiles.push_back(fast_ind);
    }
    // iterate over the fast index tiles
    for(auto &tile : fast_ind_tiles) {
      current_tiling = tile_one;
      current_tiling[0] = tile;
      index_t second_tile = max_tile / tile;
      // iterate over the second index tiles
      while(second_tile > 1) {
        current_tiling[1] = second_tile;
        // try 1 or 2 for the next two indices
        if(max_tile / tile / second_tile >=4 ){
          for(index_t i : {2, 1}) {
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
            printf("Current Tile size: %d %d %d %d, time: %11.4e\n", 
                    current_tiling[0], current_tiling[1], 
                    current_tiling[2], current_tiling[3], min_time);
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
            printf("Current Tile size: %d %d %d %d, time: %11.4e\n", 
                    current_tiling[0], current_tiling[1], 
                    current_tiling[2], current_tiling[3], min_time);
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
          printf("Current Tile size: %d %d %d %d, time: %11.4e\n", 
                  current_tiling[0], current_tiling[1], 
                  current_tiling[2], current_tiling[3], min_time);
        }
        second_tile = second_tile / 2;
      }
    }
  }
  printf("Best Tile size: %d %d %d %d\n", best_tiling[0], best_tiling[1], best_tiling[2], best_tiling[3]);
  printf("Best Time: %11.4e s\n", best_time);
  double time_rec = std::numeric_limits<double>::max();
  auto tune_policy = Policy<rank>(start, end);
  for(int ii = 0; ii < STREAM_NTIMES; ii++) {
    timer.reset();
    Kokkos::parallel_for(tune_policy, functor);
    Kokkos::fence();
    time_rec = std::min(time_rec, timer.seconds());
  }
  printf("Time with default tile size: %11.4e s\n", time_rec);
  printf("Speedup: %f\n", time_rec / best_time);
  return best_tiling;
}