#include "pipeline_template.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cppcoro/when_all.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <ranges>
#include <string>

/**
 * Section 4.1.2. GPU-Stage Mapping using divide and conquer algorithm.
 * The divide and conquer is accelerated using multithreading
 * and memoization.
 */

namespace oobleck {

map<DCExecutionResult::key, DCExecutionResult>
    PipelineTemplateGenerator::dc_cache_;
cppcoro::static_thread_pool PipelineTemplateGenerator::thread_pool_;

std::vector<LayerExecutionResult>
PipelineTemplateGenerator::get_profiler_results(const std::string& model_name,
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
  //   assert(mb.size() == allreduce_across_nodes.size() ==
  //          allreduce_in_node.size());

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

    // std::tuple<int, int> mem_required_tuple = std::make_tuple(
    //     static_cast<int>(mb[i]["mem_required"][0].get<double>()),
    //     static_cast<int>(mb[i]["mem_required"][1].get<double>()));

    layer_execution_results.emplace_back(LayerExecutionResult(
        i, mb[i]["forward"].get<double>(), mb[i]["backward"].get<double>(),
        allreduce_in_node_map, allreduce_across_nodes_map,
        mb[i]["mem_required"].get<std::tuple<int, int>>()));
  }

  std::cout << "Returning from get_profiler_results" << std::endl;

  return std::move(layer_execution_results);
}

std::vector<PipelineTemplate>
PipelineTemplateGenerator::create_pipeline_templates(
    const std::string& model_name,
    const std::string& model_tag,
    const int microbatch_size,
    const std::tuple<int, int> num_nodes,
    const int num_gpus_per_node) {
  int min_num_nodes = std::get<0>(num_nodes);
  int max_num_nodes = std::get<1>(num_nodes);

  // Load JSON files to create std::vector<LayerExecutionResult>
  std::vector<LayerExecutionResult> layer_execution_results =
      get_profiler_results(model_name, model_tag, microbatch_size);

  std::map<int, std::vector<cppcoro::task<DCExecutionResult>>> tasks;
  for (int i = min_num_nodes; i < max_num_nodes; i++) {
    std::cout << "Creating tasks for " << i << " nodes" << std::endl;
    int min_num_stages = i;
    int max_num_stages = layer_execution_results.size();
    std::vector<cppcoro::task<DCExecutionResult>> num_node_tasks;
    for (int num_stages = min_num_stages; num_stages < max_num_stages;
         num_stages++) {
      num_node_tasks.emplace_back(divide_and_conquer(
          layer_execution_results, num_stages, i, num_gpus_per_node));
    }
    tasks[i] = std::move(num_node_tasks);
  }

  std::vector<PipelineTemplate> pipeline_templates;

  for (auto& num_node_tasks : tasks) {
    std::cout << "Waiting for tasks for " << num_node_tasks.first << " nodes"
              << std::endl;
    std::vector<DCExecutionResult> results =
        cppcoro::sync_wait(cppcoro::when_all(std::move(num_node_tasks.second)));
    std::cout << "Wait done" << std::endl;

    if (std::all_of(results.begin(), results.end(),
                    [](const DCExecutionResult& result) -> bool {
                      return result.get_t() == 0.0;
                    })) {
      std::cout << "All results are 0.0" << std::endl;
      continue;
    }

    auto optimal_result = std::min_element(
        std::begin(results), std::end(results),
        [](const DCExecutionResult& a, const DCExecutionResult& b) {
          return a.get_t() < b.get_t();
        });
    std::cout << "Finding minimum element done" << std::endl;

    pipeline_templates.emplace_back(PipelineTemplate(
        optimal_result->get_stages(), layer_execution_results.size(),
        num_node_tasks.first, num_gpus_per_node));
  }

  return std::move(pipeline_templates);
}

/**
 * Use cache to map GPUs and stages using divide and conquer.
 */
cppcoro::task<DCExecutionResult> PipelineTemplateGenerator::divide_and_conquer(
    const std::vector<LayerExecutionResult>& layer_execution_results,
    const int num_stages,
    const int num_nodes,
    const int num_gpus_per_node) {
  int start_layer_index = 0;
  int end_layer_index =
      layer_execution_results[layer_execution_results.size() - 1].layer_index_;

  // Infeasible cases
  DCExecutionResult result;
  DCExecutionResult::key key =
      std::make_tuple(num_stages, start_layer_index, end_layer_index, num_nodes,
                      num_gpus_per_node);

  if (num_stages > end_layer_index - start_layer_index) {
    // If the number of stages is more than number of layers
    dc_cache_[key] = result;
    co_return result;
  }

  if (num_nodes == 1) {
    if (num_gpus_per_node < num_stages) {
      // At least one GPU should be assigned to each stage
      dc_cache_[key] = result;
      co_return result;
    }

    double log_num_gpus_per_node = log2(num_gpus_per_node);
    if (num_stages == 1 &&
        log_num_gpus_per_node != trunc(log_num_gpus_per_node)) {
      // One stage cannot have non-power-of-two number of GPUs
      dc_cache_[key] = result;
      co_return result;
    }
  } else if (num_nodes > num_stages) {
    // Two ore more node cannot be assigned to the same stage
    dc_cache_[key] = result;
    co_return result;
  }

  // Base case (conquer phase)
  if (num_stages == 1) {
    assert(num_nodes == 1);
    // If there is only one stage, assign all layers to that stage
    StageExecutionResult stage(layer_execution_results, num_gpus_per_node);
    DCExecutionResult result(stage, num_nodes, num_gpus_per_node);
    dc_cache_[result.get_key()] = result;
    co_return result;
  }

  // Divide phase
  for (int k :
       std::ranges::iota_view<int, int>(1, layer_execution_results.size())) {
    if (num_nodes == 1) {
      // Split GPUs in a node
      for (int num_gpus_left :
           std::ranges::iota_view<int, int>(1, num_gpus_per_node)) {
        for (int num_stages_left :
             std::ranges::iota_view<int, int>(1, num_stages)) {
          auto key_left = std::make_tuple(num_stages_left, start_layer_index, k,
                                          num_nodes, num_gpus_left);
          auto cache = dc_cache_.find(key_left);

          DCExecutionResult result_left;
          if (cache != dc_cache_.end()) {
            result_left = cache->second;
          } else {
            result_left = co_await divide_and_conquer(
                std::vector<LayerExecutionResult>(
                    layer_execution_results.begin(),
                    layer_execution_results.begin() + k),
                num_stages_left, num_nodes, num_gpus_left);
          }

          auto key_right =
              std::make_tuple(num_stages - num_stages_left, k, end_layer_index,
                              num_nodes, num_gpus_per_node - num_gpus_left);
          cache = dc_cache_.find(key_right);

          DCExecutionResult result_right;
          if (cache != dc_cache_.end()) {
            result_right = cache->second;
          } else {
            result_right = co_await divide_and_conquer(
                std::vector<LayerExecutionResult>(
                    layer_execution_results.begin() + k,
                    layer_execution_results.end()),
                num_stages - num_stages_left, num_nodes,
                num_gpus_per_node - num_gpus_left);
          }

          DCExecutionResult new_result(result_left, result_right, num_nodes,
                                       num_gpus_per_node);
          if (new_result.get_t() < result.get_t()) {
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
          auto key_left = std::make_tuple(num_stages_left, start_layer_index, k,
                                          num_nodes_left, num_gpus_per_node);
          auto cache = dc_cache_.find(key_left);

          DCExecutionResult result_left;
          if (cache != dc_cache_.end()) {
            result_left = cache->second;
          } else {
            result_left = co_await divide_and_conquer(
                std::vector<LayerExecutionResult>(
                    layer_execution_results.begin(),
                    layer_execution_results.begin() + k),
                num_stages_left, num_nodes_left, num_gpus_per_node);
          }

          auto key_right =
              std::make_tuple(num_stages - num_stages_left, k, end_layer_index,
                              num_nodes - num_nodes_left, num_gpus_per_node);
          cache = dc_cache_.find(key_right);

          DCExecutionResult result_right;
          if (cache != dc_cache_.end()) {
            result_right = cache->second;
          } else {
            result_right = co_await divide_and_conquer(
                std::vector<LayerExecutionResult>(
                    layer_execution_results.begin() + k,
                    layer_execution_results.end()),
                num_stages - num_stages_left, num_nodes - num_nodes_left,
                num_gpus_per_node);
          }

          DCExecutionResult new_result(result_left, result_right, num_nodes,
                                       num_gpus_per_node);
          if (new_result.get_t() < result.get_t()) {
            result = new_result;
          }
        }
      }
    }
  }

  dc_cache_[result.get_key()] = result;
  co_return result;
}

}  // namespace oobleck