#include <parallel_hashmap/phmap.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cassert>
#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <nlohmann/json.hpp>
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

void push_task(map<DCExecutionResult::key, double>& cache,
               int num_stages,
               int layer_start_index,
               int layer_end_index,
               int num_nodes,
               int num_gpus_per_node) {
  // base case
  if (num_stages == 1) {
    assert(num_nodes == 1);
  }
}

std::vector<PipelineTemplate> create_pipeline_templates(
    int f,
    int num_nodes,
    int num_gpus_per_node,
    const std::string& model_name,
    int microbatch_size) {
  // 1. Create a cache, where the key is the DCExecutionResult::key
  // and the value is the execution time.
  map<DCExecutionResult::key, double> cache;

  return {};
}

int test_task_execute() {
  cppcoro::static_thread_pool pool;

  auto job = [&]() -> cppcoro::task<int> {
    // First schedule the coroutine onto the threadpool.
    co_await pool.schedule();

    // When it resumes, this coroutine is now running on the threadpool.
    co_return 42;
  };

  return cppcoro::sync_wait(job());
}

}  // namespace oobleck

PYBIND11_MODULE(cplanning, m) {
  m.doc() = "Oobleck planning module";
  m.def("test", &oobleck::test_task_execute, "Test function");
}