#include <parallel_hashmap/phmap.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cassert>
#include <cmath>
#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <nlohmann/json.hpp>
#include <ranges>
#include <tuple>
#include "execution_result.h"
#include "pipeline_template.h"

namespace py = pybind11;
/**
 * Section 4.1.2. GPU-Stage Mapping using divide and conquer algorithm.
 * The divide and conquer is accelerated using multithreading
 * and memoization.
 */

namespace oobleck {

using LayerIndex = int;

// static moodycamel::ConcurrentQueue<DCExecutionResult> divide_queue;
// static moodycamel::ConcurrentQueue<DCExecutionResult> conquer_queue;
// static moodycamel::ConcurrentQueue<DCExecutionResult> combine_queue;
static map<LayerIndex, LayerExecutionResult> layer_execution_results;
static map<DCExecutionResult::key, double> cache;

cppcoro::task<DCExecutionResult> run_divide_and_conquer(
    int num_stages,
    std::tuple<int, int> layer_indices,
    int num_nodes,
    int num_gpus_per_node) {
  int start_layer_index = std::get<0>(layer_indices);
  int end_layer_index = std::get<1>(layer_indices);

  DCExecutionResult result;
  DCExecutionResult::key key =
      std::make_tuple(num_stages, start_layer_index, end_layer_index, num_nodes,
                      num_gpus_per_node);

  // Filter infeasible cases
  if (num_stages > end_layer_index - start_layer_index) {
    // If the number of stages is more than number of layers
    cache[key] = -1;
    co_return result;
  }

  if (num_nodes == 1) {
    if (num_gpus_per_node < num_stages) {
      // At least one GPU should be assigned to each stage
      cache[key] = -1;
      co_return result;
    }

    double log_num_gpus_per_node = log2(num_gpus_per_node);
    if (num_stages == 1 &&
        log_num_gpus_per_node != trunc(log_num_gpus_per_node)) {
      // One stage cannot have non-power-of-two number of GPUs
      cache[key] = -1;
      co_return result;
    } else if (num_nodes > num_stages) {
      // Two ore more node cannot be assigned to the same stage
      cache[key] = -1;
      co_return result;
    }
  }

  std::vector<LayerExecutionResult> layers(end_layer_index - start_layer_index);
  for (int i = start_layer_index; i < end_layer_index; i++) {
    layers[i - start_layer_index] = layer_execution_results[i];
  }

  /**
   * Base case
   */
  if (num_stages == 1) {
    assert(num_nodes == 1);
    StageExecutionResult stage(layers, num_gpus_per_node);
    DCExecutionResult result(stage, num_nodes, num_gpus_per_node);
    cache[result.get_key()] = result.get_t();
    co_return result;
  }

  /**
   * Divide and combine
   */
  for (int k : std::views::iota(1, layers.size())) {
    if (num_nodes == 1) {
      // Split GPUs in a node
      for (int num_gpus_left : std::views::iota(1, num_gpus_per_node)) {
        for (int num_stages_left : std::views::iota(1, num_stages)) {
          DCExecutionResult result_left = co_await run_divide_and_conquer(
              num_stages_left,
              std::make_tuple(start_layer_index, start_layer_index + k),
              num_nodes, num_gpus_left);
          DCExecutionResult result_right = co_await run_divide_and_conquer(
              num_stages - num_stages_left,
              std::make_tuple(start_layer_index + k, end_layer_index),
              num_nodes, num_gpus_per_node - num_gpus_left);

          if (result_left.get_status() ==
                  DCExecutionResult::Status::NOT_VALID ||
              result_right.get_status() ==
                  DCExecutionResult::Status::NOT_VALID) {
            continue;
          }

          DCExecutionResult new_result(result_left, result_right, num_nodes,
                                       num_gpus_per_node);
          if (new_result.get_t() < result.get_t()) {
            result = new_result;
          }
        }
      }
    } else {
      for (int num_nodes_left : std::views::iota(1, num_nodes)) {
        for (int num_stages_left : std::views::iota(1, num_stages)) {
          DCExecutionResult result_left = co_await run_divide_and_conquer(
              num_stages_left,
              std::make_tuple(start_layer_index, start_layer_index + k),
              num_nodes_left, num_gpus_per_node);
          DCExecutionResult result_right = co_await run_divide_and_conquer(
              num_stages - num_stages_left,
              std::make_tuple(start_layer_index + k, end_layer_index),
              num_nodes - num_nodes_left, num_gpus_per_node);

          if (result_left.get_status() ==
                  DCExecutionResult::Status::NOT_VALID ||
              result_right.get_status() ==
                  DCExecutionResult::Status::NOT_VALID) {
            continue;
          }

          DCExecutionResult new_result(result_left, result_right, num_nodes,
                                       num_gpus_per_node);
          if (new_result.get_t() < result.get_t()) {
            result = new_result;
          }
        }
      }
    }

    cache[result.get_key()] = result.get_t();
    co_return result;
  }
}

std::vector<PipelineTemplate> create_pipeline_templates(
    int f,
    int num_nodes,
    int num_gpus_per_node,
    int min_template_num_nodes,
    const std::string& model_name,
    int microbatch_size) {
  // 1. Create a cache, where the key is the DCExecutionResult::key
  // and the value is the execution time.
  map<DCExecutionResult::key, double> cache;

  int max_template_num_nodes = num_nodes - f * min_template_num_nodes;

  cppcoro::static_thread_pool pool;

  auto task = [&]() -> cppcoro::task<PipelineTemplate> {
    // First schedule the coroutine onto the threadpool.
    co_await pool.schedule();

    // When it resumes, this coroutine is now running on the threadpool.
    DCExecutionResult result = co_await run_divide_and_conquer(
        f, std::make_tuple(0, layer_execution_results.size()),
        max_template_num_nodes, num_gpus_per_node);

    PipelineTemplate ptemplate(model_name, num_nodes, num_gpus_per_node);
    ptemplate.map_gpus_and_stages(cache);

    co_return ptemplate;
  };

  PipelineTemplate ptemplate = cppcoro::sync_wait(task());

  return {ptemplate};
}

}  // namespace oobleck

PYBIND11_MODULE(cplanning, m) {
  m.doc() = "Oobleck planning module";
  // m.def("test", &oobleck::test_task_execute, "Test function");
}