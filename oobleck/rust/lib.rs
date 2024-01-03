mod execution_result;
mod pipeline_template_generator;

use pyo3::prelude::*;

#[pymodule]
fn planner(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        pipeline_template_generator::create_pipeline_templates,
        m
    )?)?;
    Ok(())
}
