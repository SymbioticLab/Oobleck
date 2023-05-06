#include "pipeline_template.h"
#include <chrono>
#include <iostream>
#include <tuple>

using namespace oobleck;
using namespace std;

int main() {
  PipelineTemplateGenerator generator;
  auto start = std::chrono::high_resolution_clock::now();
  auto templates = generator.create_pipeline_templates(
      "gpt2", "2.7b", 2, std::make_tuple(3, 12), 1);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << "ms" << std::endl;
  return 0;
}