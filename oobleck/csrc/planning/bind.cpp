#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pipeline_template.h"

namespace py = pybind11;

PYBIND11_MODULE(pipeline_template, m) {
  m.doc() = "Oobleck pipeline template module";

  py::class_<oobleck::StageExecutionResult,
             std::shared_ptr<oobleck::StageExecutionResult>>(
      m, "StageExecutionResult");

  py::class_<oobleck::PipelineTemplate>(m, "PipelineTemplate")
      .def("get_iteration_time", &oobleck::PipelineTemplate::get_iteration_time)
      .def("get_stages", &oobleck::PipelineTemplate::get_stages)
      .def("get_num_nodes", &oobleck::PipelineTemplate::get_num_nodes)
      .def("get_num_gpus_per_node",
           &oobleck::PipelineTemplate::get_num_gpus_per_node);
  //  .def(
  //      py::init<std::vector<std::shared_ptr<oobleck::StageExecutionResult>>&,
  //               int, int, int>(),
  //      py::arg("stage_execution_results"), py::arg("num_layers"),
  //      py::arg("num_nodes"), py::arg("num_gpus_per_node"))

  py::class_<oobleck::PipelineTemplateGenerator>(m, "PipelineTemplateGenerator")
      .def(py::init<>())
      .def("create_pipeline_templates",
           &oobleck::PipelineTemplateGenerator::create_pipeline_templates,
           py::arg("model_name"), py::arg("model_tag"),
           py::arg("microbatch_size"), py::arg("num_nodes"),
           py::arg("num_gpus_per_node"));
}