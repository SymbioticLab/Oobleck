#include <concurrentqueue.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <BS_thread_pool.hpp>
#include <nlohmann/json.hpp>
#include "execution_result.h"

/**
 * Section 4.1.2. GPU-Stage Mapping using divide and conquer algorithm.
 * The divide and conquer is accelerated using multithreading
 * and memoization.
 */

namespace oobleck {

int test() {
  return 0;
}

}  // namespace oobleck

namespace py = pybind11;

PYBIND11_MODULE(cplanning, m) {
  m.doc() = "Oobleck planning module";
  m.def("test", &oobleck::test, "Test function");
}