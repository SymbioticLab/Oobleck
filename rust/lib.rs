use crate::pipeline_template_generator::PipelineTemplateGenerator;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fmt;

mod execution_result;
mod pipeline_template_generator;

#[derive(Clone, Debug, PartialEq, Eq)]
enum PlannerError {
    InvalidInput(String),
    Infeasible(String),
}

impl PlannerError {
    fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    fn infeasible(message: impl Into<String>) -> Self {
        Self::Infeasible(message.into())
    }
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "invalid planner input: {message}"),
            Self::Infeasible(message) => write!(f, "pipeline planning is infeasible: {message}"),
        }
    }
}

impl std::error::Error for PlannerError {}

impl From<PlannerError> for PyErr {
    fn from(error: PlannerError) -> Self {
        match error {
            PlannerError::InvalidInput(message) => PyValueError::new_err(message),
            PlannerError::Infeasible(message) => PyRuntimeError::new_err(message),
        }
    }
}

#[pyfunction(
    signature = (
        model_name,
        profile_data,
        resource_counts,
        tensor_parallel_size=1,
        device_memory_bytes=None
    )
)]
fn create_pipeline_templates(
    py: Python<'_>,
    model_name: String,
    profile_data: Vec<execution_result::LayerExecutionResult>,
    mut resource_counts: Vec<u32>,
    tensor_parallel_size: u32,
    device_memory_bytes: Option<u64>,
) -> PyResult<Py<PyDict>> {
    if model_name.trim().is_empty() {
        return Err(PlannerError::invalid_input("model_name must not be empty").into());
    }
    if resource_counts.is_empty() {
        return Err(PlannerError::invalid_input("resource_counts must not be empty").into());
    }
    if resource_counts.contains(&0) {
        return Err(PlannerError::invalid_input("resource counts must be positive").into());
    }
    if tensor_parallel_size == 0 {
        return Err(PlannerError::invalid_input("tensor_parallel_size must be positive").into());
    }
    if device_memory_bytes == Some(0) {
        return Err(
            PlannerError::invalid_input("device_memory_bytes must be positive when supplied").into(),
        );
    }

    resource_counts.sort_unstable();
    resource_counts.dedup();
    let max_resource_count = *resource_counts
        .last()
        .expect("non-empty resource counts were validated");
    let mut generator = PipelineTemplateGenerator::new(profile_data)?;
    generator.plan(max_resource_count, device_memory_bytes)?;
    let results = PyDict::new(py);

    for resource_count in resource_counts {
        let result = generator.get_pipeline_template(resource_count)?;
        let template = PyDict::new(py);
        let ranges: Vec<(u32, u32)> = result.stages.iter().map(|stage| stage.layers).collect();
        template.set_item(
            "template_id",
            format!("{model_name}-stages-{resource_count}"),
        )?;
        template.set_item("layer_ranges", ranges)?;
        template.set_item("tensor_parallel_size", tensor_parallel_size)?;
        template.set_item("forward_time", result.forward_time())?;
        template.set_item("backward_time", result.backward_time())?;
        template.set_item("communication_time", 0.0)?;
        template.set_item("paper_t1", result.t1)?;
        template.set_item("paper_t3", result.t3)?;
        template.set_item("paper_bottleneck_stage", result.kstar)?;
        template.set_item("activation_memory", result.activation_memory())?;
        template.set_item("persistent_memory", result.persistent_memory())?;
        template.set_item(
            "max_microbatches",
            result.max_microbatches(device_memory_bytes),
        )?;
        template.set_item("fingerprint", py.None())?;
        template.set_item("schema_version", 1)?;
        results.set_item(resource_count, template)?;
    }

    Ok(results.unbind())
}

#[pymodule]
fn planner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_pipeline_templates, m)?)?;
    Ok(())
}
