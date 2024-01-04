use crate::pipeline_template_generator::PipelineTemplateGenerator;
mod execution_result;
mod pipeline_template_generator;
use env_logger;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

#[derive(Debug)]
struct PlannerError {
    message: String,
}

impl PlannerError {
    fn new(message: &str) -> Self {
        PlannerError {
            message: message.to_string(),
        }
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

#[pyclass]
struct PipelineTemplate {
    #[pyo3(get)]
    latency: f64,
    #[pyo3(get)]
    mem_required: u64,
    #[pyo3(get)]
    modules_per_stage: Vec<Vec<String>>,
}

#[pyfunction]
fn create_pipeline_templates(
    model_name: &str,
    tag: &str,
    mut num_nodes: Vec<u32>,
    oobleck_base_dir: Option<PathBuf>,
) -> Result<HashMap<u32, PipelineTemplate>, PlannerError> {
    num_nodes.sort();

    let mut generator = PipelineTemplateGenerator::new(model_name, tag, oobleck_base_dir);
    generator.divide_and_conquer(num_nodes[num_nodes.len() - 1])?;

    let mut results: HashMap<u32, PipelineTemplate> = HashMap::new();
    for num_node in num_nodes {
        let template = generator.get_pipeline_template(num_node)?;
        results.insert(num_node, template);
    }

    Ok(results)
}

#[pymodule]
fn planner(_py: Python, m: &PyModule) -> PyResult<()> {
    let _ = env_logger::try_init();
    m.add_class::<PipelineTemplate>()?;
    m.add_function(wrap_pyfunction!(create_pipeline_templates, m)?)?;
    Ok(())
}
