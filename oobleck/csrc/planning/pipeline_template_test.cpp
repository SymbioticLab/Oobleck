#include "pipeline_template.h"
#include <tuple>

using namespace oobleck;
using namespace std;

int main() {
  PipelineTemplateGenerator generator;
  auto templates = generator.create_pipeline_templates(
      "gpt2", "2.7b", 2, std::make_tuple(2, 6), 1);
  return 0;
}