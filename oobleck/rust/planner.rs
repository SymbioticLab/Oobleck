use pyo3::prelude::*;

#[pyfunction]
fn test() -> PyResult<()> {
    println!("Hello, world!");
    Ok(())
}