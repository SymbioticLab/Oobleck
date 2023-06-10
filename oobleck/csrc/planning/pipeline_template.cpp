#include "pipeline_template.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cppcoro/when_all.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <ranges>
#include <string>

#ifdef PYBIND11_MODULE
#include <pybind11/pybind11.h>
#endif

/**
 * Section 4.1.2. GPU-Stage Mapping using divide and conquer algorithm.
 * The divide and conquer is accelerated using multithreading
 * and memoization.
 */

namespace oobleck {

// CacheMap PipelineTemplateGenerator::dc_cache_;
// cppcoro::static_thread_pool PipelineTemplateGenerator::thread_pool_;

std::shared_ptr<LayerExecutionResults> get_profiler_results(
    const std::string& model_name,
    const std::string& model_tag,
    const int microbatch_size) {
  auto get_cache = [](const std::string& cache_path) -> nlohmann::json {
    std::ifstream ifs(cache_path);
    assert(ifs.is_open());
    return nlohmann::json::parse(ifs);
  };

  std::string profile_path =
      "/tmp/oobleck/profiles/" + model_name + "-" + model_tag;
  auto mb = get_cache(profile_path + "/mb" + std::to_string(microbatch_size) +
                      ".json");
  auto allreduce_in_node = get_cache(profile_path + "/allreduce_in_node.json");
  auto allreduce_across_nodes =
      get_cache(profile_path + "/allreduce_across_nodes.json");
  std::cout << "mb size: " << mb.size()
            << ", ar across: " << allreduce_across_nodes.size()
            << ", ar in: " << allreduce_in_node.size() << std::endl;
  // assert(mb.size() == allreduce_across_nodes.size() ==
  //        allreduce_in_node.size());

  std::cout << "Loading done. creating layer execution results..." << std::endl;

  int num_layers = mb.size();
  std::vector<LayerExecutionResult> layer_execution_results;
  for (int i = 0; i < num_layers; i++) {
    std::map<int, double> allreduce_in_node_map;
    for (auto& [key, value] : allreduce_in_node[i].items()) {
      allreduce_in_node_map[std::stoi(key)] = value;
    }
    std::map<int, double> allreduce_across_nodes_map;
    for (auto& [key, value] : allreduce_across_nodes[i].items()) {
      allreduce_across_nodes_map[std::stoi(key)] = value;
    }

    layer_execution_results.emplace_back(LayerExecutionResult(
        i, mb[i]["forward"].get<double>(), mb[i]["backward"].get<double>(),
        allreduce_in_node_map, allreduce_across_nodes_map,
        mb[i]["mem_required"].get<std::tuple<int, int>>()));
  }

  std::cout << "Returning from get_profiler_results" << std::endl;

  return std::make_shared<LayerExecutionResults>(
      std::move(layer_execution_results));
}

std::vector<PipelineTemplate>
PipelineTemplateGenerator::create_pipeline_templates(
    std::shared_ptr<LayerExecutionResults> layer_execution_results,
    const std::tuple<int, int>& num_nodes,
    const int num_gpus_per_node) {
  int min_num_nodes = std::get<0>(num_nodes);
  int max_num_nodes = std::get<1>(num_nodes);
#ifdef PYBIND11_MODULE
  // Release GIL
  pybind11::gil_scoped_release release;
#endif

  std::map<int, std::vector<cppcoro::task<std::shared_ptr<DCExecutionResult>>>>
      tasks;
  for (int i = min_num_nodes; i <= max_num_nodes; i++) {
    std::cout << "Creating tasks for " << i << " nodes" << std::endl;
    int min_num_stages = i;
    int max_num_stages = layer_execution_results->size();
    std::vector<cppcoro::task<std::shared_ptr<DCExecutionResult>>>
        num_node_tasks;
    for (int num_stages = min_num_stages; num_stages <= max_num_stages;
         num_stages++) {
      num_node_tasks.emplace_back(divide_and_conquer(
          layer_execution_results,
          std::make_tuple(0, layer_execution_results->size()), num_stages, i,
          num_gpus_per_node));
    }
    tasks[i] = std::move(num_node_tasks);
  }

  std::vector<PipelineTemplate> pipeline_templates;
  // for (auto num_node_tasks = tasks.rbegin(); num_node_tasks !=
  // tasks.rend();
  // for (auto&& num_node_tasks = tasks.begin(); num_node_tasks != tasks.end();
  //      num_node_tasks++) {
  for (auto&& num_node_tasks : tasks) {
    std::cout << "Waiting for tasks for " << num_node_tasks.first << " nodes"
              << std::endl;
    std::vector<std::shared_ptr<DCExecutionResult>> results =
        cppcoro::sync_wait(cppcoro::when_all(std::move(num_node_tasks.second)));
    std::cout << "Wait done" << std::endl;

    std::cout << "Cache hit: " << cache_hit_.load()
              << ", miss: " << cache_miss_.load() << std::endl;

    if (std::all_of(results.begin(), results.end(),
                    [](const std::shared_ptr<DCExecutionResult>& result)
                        -> bool { return result == nullptr; })) {
      std::cout << "All results are invalid" << std::endl;
      continue;
    }

    auto optimal_result = [&]() -> std::shared_ptr<DCExecutionResult> {
      std::shared_ptr<DCExecutionResult> result(nullptr);
      for (int i = 0; i < results.size(); i++) {
        if (result == nullptr) {
          result = results[i];
        } else if (results[i] != nullptr &&
                   results[i]->get_t() < result->get_t()) {
          result = results[i];
        }
      }
      return result;
    }();

    assert(optimal_result != nullptr &&
           optimal_result->get_stages().size() > 0);
    pipeline_templates.emplace_back(
        PipelineTemplate(optimal_result->get_stages(), optimal_result->get_t(),
                         layer_execution_results->size(), num_node_tasks.first,
                         num_gpus_per_node));
  }

#ifdef PYBIND11_MODULE
  // Acquire GIL
  pybind11::gil_scoped_acquire acquire;
#endif

  return pipeline_templates;
}

/**
 * Use cache to map GPUs and stages using divide and conquer.
 */
cppcoro::task<std::shared_ptr<DCExecutionResult>>
PipelineTemplateGenerator::divide_and_conquer(
    std::shared_ptr<LayerExecutionResults> layer_execution_results,
    const std::tuple<int, int> layer_indices,
    const int num_stages,
    const int num_nodes,
    const int num_gpus_per_node) {
  co_await thread_pool_.schedule();

  int start_layer_index = std::get<0>(layer_indices);
  int end_layer_index = std::get<1>(layer_indices);

  std::shared_ptr<DCExecutionResult> result(nullptr);
  DCExecutionResult::key key =
      std::make_tuple(num_stages, start_layer_index, end_layer_index, num_nodes,
                      num_gpus_per_node);

  // Return cached result if it exists
  auto it = dc_cache_.find(key);
  if (it != dc_cache_.end()) {
    cache_hit_.fetch_add(1, std::memory_order_relaxed);
    result = it->second;
    co_return result;
  }

  cache_miss_.fetch_add(1, std::memory_order_relaxed);

  // Infeasible cases
  bool infeasible = false;
  if (num_stages > end_layer_index - start_layer_index) {
    // If the number of stages is more than number of layers
    infeasible = true;
  }

  if (num_nodes == 1) {
    if (num_gpus_per_node < num_stages) {
      // At least one GPU should be assigned to each stage
      infeasible = true;
    }

    double log_num_gpus_per_node = log2(num_gpus_per_node);
    if (num_stages == 1 &&
        log_num_gpus_per_node != trunc(log_num_gpus_per_node)) {
      infeasible = true;
    }
  } else if (num_nodes > num_stages) {
    // Two or more node cannot be assigned to the same stage
    infeasible = true;
  }

  if (infeasible) {
    dc_cache_.insert({key, nullptr});
    // accessor->second = nullptr;
    co_return nullptr;
  }

  // Base case (conquer phase)
  if (num_stages == 1) {
    assert(num_nodes == 1);
    // If there is only one stage, assign all layers to that stage
    auto stage = std::make_shared<StageExecutionResult>(
        layer_execution_results, layer_indices, num_gpus_per_node);
    auto result = std::make_shared<DCExecutionResult>(stage, num_nodes,
                                                      num_gpus_per_node);
    dc_cache_.insert({key, result});
    // accessor->second = result;
    co_return result;
  }

  // Divide phase
  for (int k : std::ranges::iota_view<int, int>(start_layer_index + 1,
                                                end_layer_index)) {
    if (num_nodes == 1) {
      // Split GPUs in a node
      for (int num_gpus_left :
           std::ranges::iota_view<int, int>(1, num_gpus_per_node)) {
        for (int num_stages_left :
             std::ranges::iota_view<int, int>(1, num_stages)) {
          std::shared_ptr<DCExecutionResult> result_left(nullptr);
          std::shared_ptr<DCExecutionResult> result_right(nullptr);

          auto key_left = std::make_tuple(num_stages_left, start_layer_index, k,
                                          num_nodes, num_gpus_left);

          auto it = dc_cache_.find(key_left);
          if (it != dc_cache_.end()) {
            result_left = it->second;
          } else {
            result_left = co_await divide_and_conquer(
                layer_execution_results, std::make_tuple(start_layer_index, k),
                num_stages_left, num_nodes, num_gpus_left);
          }

          auto key_right =
              std::make_tuple(num_stages - num_stages_left, k, end_layer_index,
                              num_nodes, num_gpus_per_node - num_gpus_left);

          it = dc_cache_.find(key_right);
          if (it != dc_cache_.end()) {
            result_right = it->second;
          } else {
            result_right = co_await divide_and_conquer(
                layer_execution_results, std::make_tuple(k, end_layer_index),
                num_stages - num_stages_left, num_nodes,
                num_gpus_per_node - num_gpus_left);
          }

          if (result_left == nullptr || result_right == nullptr) {
            continue;
          }

          auto new_result = std::make_shared<DCExecutionResult>(
              result_left, result_right, num_nodes, num_gpus_per_node);
          if (result == nullptr || new_result->get_t() < result->get_t()) {
            result = new_result;
          }
        }
      }
    } else {
      // Split nodes
      for (int num_nodes_left :
           std::ranges::iota_view<int, int>(1, num_nodes)) {
        for (int num_stages_left :
             std::ranges::iota_view<int, int>(1, num_stages)) {
          std::shared_ptr<DCExecutionResult> result_left(nullptr);
          std::shared_ptr<DCExecutionResult> result_right(nullptr);

          auto key_left = std::make_tuple(num_stages_left, start_layer_index, k,
                                          num_nodes_left, num_gpus_per_node);

          auto it = dc_cache_.find(key_left);
          if (it != dc_cache_.end()) {
            result_left = it->second;
          } else {
            result_left = co_await divide_and_conquer(
                layer_execution_results, std::make_tuple(start_layer_index, k),
                num_stages_left, num_nodes_left, num_gpus_per_node);
          }

          auto key_right =
              std::make_tuple(num_stages - num_stages_left, k, end_layer_index,
                              num_nodes - num_nodes_left, num_gpus_per_node);

          it = dc_cache_.find(key_right);
          if (it != dc_cache_.end()) {
            result_right = it->second;
          } else {
            result_right = co_await divide_and_conquer(
                layer_execution_results, std::make_tuple(k, end_layer_index),
                num_stages - num_stages_left, num_nodes - num_nodes_left,
                num_gpus_per_node);
          }

          if (result_left == nullptr || result_right == nullptr) {
            continue;
          }

          auto new_result = std::make_shared<DCExecutionResult>(
              result_left, result_right, num_nodes, num_gpus_per_node);
          if (result == nullptr || new_result->get_t() < result->get_t()) {
            result = new_result;
          }
        }
      }
    }
  }

  dc_cache_.insert({key, result});
  co_return result;
}

}  // namespace oobleck