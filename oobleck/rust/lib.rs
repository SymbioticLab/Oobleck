mod execution_result;
mod pipeline_template_generator;

use pyo3::prelude::*;

#[pymodule]
fn planner(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<pipeline_template_generator::PipelineTemplateGenerator>()?;
    Ok(())
}
