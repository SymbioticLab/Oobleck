#include "pipeline_template.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cppcoro/when_all.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <ranges>
#include <string>

/**
 * Section 4.1.2. GPU-Stage Mapping using divide and conquer algorithm.
 * The divide and conquer is accelerated using multithreading
 * and memoization.
 */

namespace oobleck {

map<DCExecutionResult::key, std::shared_ptr<DCExecutionResult>>
    PipelineTemplateGenerator::dc_cache_;
cppcoro::static_thread_pool PipelineTemplateGenerator::thread_pool_;

std::unique_ptr<std::vector<LayerExecutionResult>>
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
  std::unique_ptr<std::vector<LayerExecutionResult>> layer_execution_results;
  // std::vector<LayerExecutionResult> layer_execution_results;
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

    layer_execution_results->emplace_back(LayerExecutionResult(
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
  auto layer_execution_results =
      get_profiler_results(model_name, model_tag, microbatch_size);

  /**
   * TESTTESTTEESTTESTTESTTEESTTESTTESTTEESTTESTTESTTEEST
   */
  {
    std::vector<cppcoro::task<std::shared_ptr<DCExecutionResult>>> tasks;
    for (int i = min_num_nodes; i < max_num_nodes; i++) {
      int min_num_stages = i;
      int max_num_stages = layer_execution_results->size();
      for (int num_stages = min_num_stages; num_stages < max_num_stages;
           num_stages++) {
        tasks.emplace_back(
            divide_and_conquer(std::move(layer_execution_results), num_stages,
                               i, num_gpus_per_node));
      }
    }

    std::vector<std::shared_ptr<DCExecutionResult>> results =
        cppcoro::sync_wait(cppcoro::when_all(std::move(tasks)));
    std::cout << "Wait done" << std::endl;

    std::cout << "Cache hit: " << cache_hit_.load()
              << ", miss: " << cache_miss_.load() << std::endl;
    exit(1);
  }

  // =====================================================
#if 0
  std::map<int, std::vector<cppcoro::task<DCExecutionResult>>> tasks;
  for (int i = min_num_nodes; i < max_num_nodes; i++) {
    std::cout << "Creating tasks for " << i << " nodes" << std::endl;
    int min_num_stages = i;
    int max_num_stages = layer_execution_results->size();
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

    std::cout << "Cache hit: " << cache_hit_.load()
              << ", miss: " << cache_miss_.load() << std::endl;

    if (std::all_of(results.begin(), results.end(),
                    [](const DCExecutionResult& result) -> bool {
                      return result.get_t() ==
                             std::numeric_limits<double>::infinity();
                    })) {
      std::cout << "All results are invalid" << std::endl;
      continue;
    }

    auto optimal_result = std::min_element(
        std::begin(results), std::end(results),
        [](const DCExecutionResult& a, const DCExecutionResult& b) {
          return a.get_t() < b.get_t();
        });
    std::cout << "Finding minimum element done" << std::endl;

    assert(optimal_result != std::end(results) &&
           optimal_result->get_stages().size() > 0);
    pipeline_templates.emplace_back(PipelineTemplate(
        optimal_result->get_stages(), layer_execution_results->size(),
        num_node_tasks.first, num_gpus_per_node));
  }

  return std::move(pipeline_templates);
#endif
}

/**
 * Use cache to map GPUs and stages using divide and conquer.
 */
cppcoro::task<std::shared_ptr<DCExecutionResult>>
PipelineTemplateGenerator::divide_and_conquer(
    std::unique_ptr<std::vector<LayerExecutionResult>> layer_execution_results,
    const int num_stages,
    const int num_nodes,
    const int num_gpus_per_node) {
  co_await thread_pool_.schedule();

  int start_layer_index = 0;
  int end_layer_index = layer_execution_results->back().layer_index_;

  std::shared_ptr<DCExecutionResult> result(nullptr);
  DCExecutionResult::key key =
      std::make_tuple(num_stages, start_layer_index, end_layer_index, num_nodes,
                      num_gpus_per_node);

  // Return cached result if it exists
  if (dc_cache_.find(key) != dc_cache_.end()) {
    cache_hit_++;
    dc_cache_.if_contains(
        key,
        [&result](
            const map<DCExecutionResult::key,
                      std::shared_ptr<DCExecutionResult>>::value_type& value) {
          result = value.second;
        });
    co_return result;
  }

  cache_miss_++;

  // Infeasible cases
  if (num_stages > end_layer_index - start_layer_index) {
    // If the number of stages is more than number of layers
    dc_cache_.try_emplace_l(
        key,
        [](map<DCExecutionResult::key,
               std::shared_ptr<DCExecutionResult>>::value_type&) {},
        nullptr);
    co_return nullptr;
  }

  if (num_nodes == 1) {
    if (num_gpus_per_node < num_stages) {
      // At least one GPU should be assigned to each stage
      dc_cache_.try_emplace_l(
          key,
          [](map<DCExecutionResult::key,
                 std::shared_ptr<DCExecutionResult>>::value_type&) {},
          nullptr);
      co_return nullptr;
    }

    double log_num_gpus_per_node = log2(num_gpus_per_node);
    if (num_stages == 1 &&
        log_num_gpus_per_node != trunc(log_num_gpus_per_node)) {
      // One stage cannot have non-power-of-two number of GPUs
      dc_cache_.try_emplace_l(
          key,
          [](map<DCExecutionResult::key,
                 std::shared_ptr<DCExecutionResult>>::value_type&) {},
          nullptr);
      co_return nullptr;
    }
  } else if (num_nodes > num_stages) {
    // Two ore more node cannot be assigned to the same stage
    dc_cache_.try_emplace_l(
        key,
        [](map<DCExecutionResult::key,
               std::shared_ptr<DCExecutionResult>>::value_type&) {},
        nullptr);
    co_return nullptr;
  }

  // Base case (conquer phase)
  if (num_stages == 1) {
    assert(num_nodes == 1);
    // If there is only one stage, assign all layers to that stage
    StageExecutionResult stage(std::move(layer_execution_results),
                               num_gpus_per_node);
    auto result = std::make_shared<DCExecutionResult>(
        std::move(stage), num_nodes, num_gpus_per_node);
    std::cout << "Adding cache (t): " << std::to_string(result->get_t())
              << std::endl;
    dc_cache_.try_emplace_l(
        result->get_key(),
        [](map<DCExecutionResult::key,
               std::shared_ptr<DCExecutionResult>>::value_type&) {},
        result);
    co_return result;
  }

  // Divide phase
  for (int k :
       std::ranges::iota_view<int, int>(1, layer_execution_results->size())) {
    if (num_nodes == 1) {
      // Split GPUs in a node
      for (int num_gpus_left :
           std::ranges::iota_view<int, int>(1, num_gpus_per_node)) {
        for (int num_stages_left :
             std::ranges::iota_view<int, int>(1, num_stages)) {
          auto key_left = std::make_tuple(num_stages_left, start_layer_index, k,
                                          num_nodes, num_gpus_left);
          auto cache = dc_cache_.find(key_left);

          std::shared_ptr<DCExecutionResult> result_left(nullptr);
          std::shared_ptr<DCExecutionResult> result_right(nullptr);

          if (cache != dc_cache_.end()) {
            result_left = cache->second;
          } else {
            result_left = co_await divide_and_conquer(
                std::make_unique<std::vector<LayerExecutionResult>>(
                    layer_execution_results->begin(),
                    layer_execution_results->begin() + k),
                num_stages_left, num_nodes, num_gpus_left);
          }

          auto key_right =
              std::make_tuple(num_stages - num_stages_left, k, end_layer_index,
                              num_nodes, num_gpus_per_node - num_gpus_left);
          cache = dc_cache_.find(key_right);

          if (cache != dc_cache_.end()) {
            result_right = cache->second;
          } else {
            result_right = co_await divide_and_conquer(
                std::make_unique<std::vector<LayerExecutionResult>>(
                    layer_execution_results->begin() + k,
                    layer_execution_results->end()),
                num_stages - num_stages_left, num_nodes,
                num_gpus_per_node - num_gpus_left);
          }

          auto new_result = std::make_shared<DCExecutionResult>(
              result_left, result_right, num_nodes, num_gpus_per_node);
          if (new_result->get_t() < result->get_t()) {
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

          std::shared_ptr<DCExecutionResult> result_left(nullptr);
          std::shared_ptr<DCExecutionResult> result_right(nullptr);

          if (cache != dc_cache_.end()) {
            result_left = cache->second;
          } else {
            result_left = co_await divide_and_conquer(
                std::make_unique<std::vector<LayerExecutionResult>>(
                    layer_execution_results->begin(),
                    layer_execution_results->begin() + k),
                num_stages_left, num_nodes_left, num_gpus_per_node);
          }

          auto key_right =
              std::make_tuple(num_stages - num_stages_left, k, end_layer_index,
                              num_nodes - num_nodes_left, num_gpus_per_node);
          cache = dc_cache_.find(key_right);

          if (cache != dc_cache_.end()) {
            result_right = cache->second;
          } else {
            result_right = co_await divide_and_conquer(
                std::make_unique<std::vector<LayerExecutionResult>>(
                    layer_execution_results->begin() + k,
                    layer_execution_results->end()),
                num_stages - num_stages_left, num_nodes - num_nodes_left,
                num_gpus_per_node);
          }

          auto new_result = std::make_shared<DCExecutionResult>(
              result_left, result_right, num_nodes, num_gpus_per_node);
          if (new_result->get_t() < result->get_t()) {
            result = new_result;
          }
        }
      }
    }
  }

  dc_cache_.try_emplace_l(
      result->get_key(),
      [](map<DCExecutionResult::key,
             std::shared_ptr<DCExecutionResult>>::value_type&) {},
      result);
  co_return result;
}

}  // namespace oobleck