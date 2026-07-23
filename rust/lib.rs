use crate::pipeline_template_generator::PipelineTemplateGenerator;
mod execution_result;
mod pipeline_template_generator;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fmt;

#[derive(Debug)]
struct PlannerError {
    message: String,
}

impl PlannerError {
    fn new(message: &str) -> Self {
        Self { message: message.to_string() }
    }
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PlannerError: {}", self.message)
    }
}

impl std::error::Error for PlannerError {}

impl From<PlannerError> for PyErr {
    fn from(error: PlannerError) -> PyErr {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string())
    }
}

#[pyfunction(signature = (model_name, profile_data, num_nodes, tensor_parallel_size=1))]
fn create_pipeline_templates(
    py: Python<'_>,
    model_name: String,
    profile_data: Vec<execution_result::LayerExecutionResult>,
    mut num_nodes: Vec<u32>,
    tensor_parallel_size: u32,
) -> PyResult<Py<PyDict>> {
    if num_nodes.is_empty() {
        return Err(PlannerError::new("num_nodes must not be empty").into());
    }
    if tensor_parallel_size == 0 {
        return Err(PlannerError::new("tensor_parallel_size must be positive").into());
    }
    num_nodes.sort_unstable();
    num_nodes.dedup();

    let mut generator = PipelineTemplateGenerator::new(profile_data);
    generator.divide_and_conquer(*num_nodes.last().unwrap())?;
    let results = PyDict::new(py);

    for num_node in num_nodes {
        let result = generator.get_pipeline_template(num_node)?;
        let template = PyDict::new(py);
        let ranges: Vec<(u32, u32)> = result
            .stages
            .iter()
            .map(|stage| stage.layers)
            .collect();
        template.set_item("template_id", format!("{}-stages-{}", model_name, num_node))?;
        template.set_item("layer_ranges", ranges)?;
        template.set_item("tensor_parallel_size", tensor_parallel_size)?;
        template.set_item("forward_time", result.forward_time())?;
        template.set_item("backward_time", result.backward_time())?;
        template.set_item("communication_time", 0.0)?;
        template.set_item("activation_memory", result.activation_memory())?;
        template.set_item("persistent_memory", result.persistent_memory())?;
        template.set_item("max_microbatches", py.None())?;
        template.set_item("fingerprint", py.None())?;
        template.set_item("schema_version", 1)?;
        results.set_item(num_node, template)?;
    }

    Ok(results.unbind())
}

#[pymodule]
fn planner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = env_logger::try_init();
    m.add_function(wrap_pyfunction!(create_pipeline_templates, m)?)?;
    Ok(())
}
