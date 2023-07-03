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
        ranks_per_layer_(get_ranks_per_layer()) {
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

  const std::vector<int> get_pipeline_ranks(int start_rank,
                                            const int fsdp_index) const {
    if (fsdp_index < 0) {
      throw pybind11::value_error("fsdp_index must be >= 0");
    }
    if (fsdp_index >= num_gpus_per_node_) {
      throw pybind11::value_error(
          "fsdp_index must be less than num_gpus_per_node");
    }

    std::vector<int> ranks;
    for (auto& ranks_per_layer : ranks_per_layer_) {
      ranks.push_back(ranks_per_layer[fsdp_index] + start_rank);
    }

    return ranks;
  }

  const std::vector<int> get_layer_ranks(int start_rank,
                                         const int layer_index) const {
    if (layer_index >= ranks_per_layer_.size()) {
      throw pybind11::index_error(
          "layer_index must be less than the number of layers");
    }

    std::vector<int> ranks = ranks_per_layer_[layer_index];
    for (auto& rank : ranks) {
      rank += start_rank;
    }
    return ranks;
  }

 private:
  std::vector<std::shared_ptr<StageExecutionResult>> stage_execution_results_;
  const double iteration_time_;
  const int num_nodes_;
  const int num_gpus_per_node_;
  std::vector<std::vector<int>> ranks_per_layer_;

  const std::vector<std::vector<int>> get_ranks_per_layer() const {
    int max_num_gpus_in_stage = 0;
    for (auto& stage : stage_execution_results_) {
      max_num_gpus_in_stage = std::max(max_num_gpus_in_stage, stage->num_gpus_);
    }

    // Retuns a list of ranks per layer
    std::vector<std::vector<int>> ranks;
    for (auto& stage : stage_execution_results_) {
      int rank = 0;
      const int repeat_count = max_num_gpus_in_stage / stage->num_gpus_;
      std::vector<int> stage_ranks(max_num_gpus_in_stage);
      auto it = stage_ranks.begin();

      // Adjust number of ranks per layer to fix the number of GPUs
      // e.g. stage 1 uses 4 GPUs and stage 2 uses 2 GPUs:
      // ranks will be [[0, 1, 2, 3], [4, 4, 5, 5]]
      for (int i = 0; i < stage->num_gpus_; i++) {
        std::fill_n(it, repeat_count, rank);
        std::advance(it, repeat_count);
        rank++;
      }

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