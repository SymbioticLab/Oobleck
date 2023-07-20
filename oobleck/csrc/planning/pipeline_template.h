#ifndef _OOBLECK_PLANNING_PIPELINE_TEMPLATE_H_
#define _OOBLECK_PLANNING_PIPELINE_TEMPLATE_H_

#include <oneapi/tbb/concurrent_hash_map.h>
#include <pybind11/pybind11.h>
#include <atomic>
#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>
#include "execution_result.h"

namespace oobleck {

class PipelineTemplate {
 public:
  PipelineTemplate(const std::vector<std::shared_ptr<StageExecutionResult>>&
                       stage_execution_results,
                   const double iteration_time,
                   const int num_layers,
                   const int num_nodes,
                   const int num_gpus_per_node)
      : stage_execution_results_(stage_execution_results),
        iteration_time_(iteration_time),
        num_nodes_(num_nodes),
        num_gpus_per_node_(num_gpus_per_node),
        ranks_per_layer_(get_ranks_per_layer(num_gpus_per_node, num_layers)) {
    // Run divide and conquer to create a vector of StageExecutionResult
    // Perform assertion
    // 1. num_nodes * num_gpus_per_node == all # GPUs used by stage results
    // 2. stages cover all layers
    int num_gpus_used = 0;
    for (auto& stage : stage_execution_results_) {
      std::cout << stage->to_string() << std::endl;
      num_gpus_used += stage->num_gpus_;
    }
    assert(num_gpus_used == num_nodes * num_gpus_per_node);

    int stage_num_layers = 0;
    for (auto& stage : stage_execution_results_) {
      stage_num_layers += stage->num_layers();
    }
    assert(stage_num_layers == num_layers);
  }

  const double get_iteration_time() const { return iteration_time_; }
  const std::vector<std::shared_ptr<StageExecutionResult>>& get_stages() const {
    return stage_execution_results_;
  }
  int get_num_nodes() const { return num_nodes_; }
  int get_num_gpus_per_node() const { return num_gpus_per_node_; }

  std::vector<std::vector<int>> get_ranks(const int rank_offset) const {
    std::vector<std::vector<int>> ranks;
    for (auto& ranks_per_layer : ranks_per_layer_) {
      std::vector<int> ranks_per_layer_offset;
      for (auto& rank : ranks_per_layer) {
        ranks_per_layer_offset.push_back(rank + rank_offset);
      }
      ranks.push_back(ranks_per_layer_offset);
    }
    return std::move(ranks);
  }

 private:
  std::vector<std::shared_ptr<StageExecutionResult>> stage_execution_results_;
  const double iteration_time_;
  const int num_nodes_;
  const int num_gpus_per_node_;

  // ranks_per_layer is a 2D vector of ranks per layer
  // that is used by get_pipeline_ranks() and get_layer_ranks()
  // to return horizontal list of ranks or vertical, respectively.
  std::vector<std::vector<int>> ranks_per_layer_;

  // Fill the 2D grid of ranks per layer.
  // e.g. stage 1 uses 4 GPUs and stage 2 uses 2 GPUs:
  // ranks_per_layer will be [[0, 1, 2, 3], [4, 4, 5, 5]]
  // number of ranks per layer is always equals to the number of GPUs per node
  // If a stage uses less GPUs than the number of GPUs per node,
  // the ranks will be repeated.
  // Ranks provided by this function starts from 0.
  const std::vector<std::vector<int>> get_ranks_per_layer(
      const int num_gpus_per_node,
      const int num_layers) const {
    std::vector<std::vector<int>> ranks;
    int rank = 0;
    for (auto& stage : stage_execution_results_) {
      std::vector<int> stage_ranks(num_gpus_per_node);
      auto it = stage_ranks.begin();

      // Adjust number of ranks per layer to fix the number of GPUs
      // e.g. stage 1 uses 4 GPUs and stage 2 uses 2 GPUs:
      // ranks will be [[0, 1, 2, 3], [4, 4, 5, 5]]
      const int repeat_count = num_gpus_per_node / stage->num_gpus_;
      for (int i = 0; i < stage->num_gpus_; i++) {
        std::fill_n(it, repeat_count, rank);
        std::advance(it, repeat_count);
        rank++;
      }

      // push per-layer ranks to the result
      for (auto layer_index : stage->layer_indices_) {
        ranks.push_back(stage_ranks);
      }
    }

    return ranks;
  }
};

std::shared_ptr<LayerExecutionResults> get_profile_results(
    const std::string& model_name,
    const std::string& model_tag,
    const int microbatch_size);

class PipelineTemplateGenerator {
 public:
  CacheMap dc_cache_;
  cppcoro::static_thread_pool thread_pool_;

  std::vector<PipelineTemplate> create_pipeline_templates(
      std::shared_ptr<LayerExecutionResults> layer_execution_results,
      const std::tuple<int, int>& num_nodes,
      const int num_gpus_per_node);

 private:
  cppcoro::task<std::shared_ptr<DCExecutionResult>> divide_and_conquer(
      std::shared_ptr<LayerExecutionResults> layer_execution_results,
      const std::tuple<int, int> layer_indices,
      const int num_stages,
      const int num_nodes,
      const int num_gpus_per_node);

  std::atomic<unsigned long> cache_hit_;
  std::atomic<unsigned long> cache_miss_;
};

}  // namespace oobleck

#endif