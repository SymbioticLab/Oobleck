use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, PartialEq};
use std::sync::Arc;

#[derive(Serialize, Deserialize)]
pub struct ProfileResult {
    model_name: String,
    microbatch_size: u32,
    tp_size: u32,
    precision: String,
    layers: Vec<LayerExecutionResult>,
}

#[derive(Serialize, Deserialize)]
pub struct LayerExecutionResult {
    pub layer_index: u32,
    pub layer_name: String,
    pub forward: f64,
    pub backward: f64,
    pub mem_required: u64,
    pub activation_memory: u64,
    pub persistent_memory: u64,
}

impl<'py> FromPyObject<'py> for LayerExecutionResult {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let mem_required = ob.getattr("mem_required")?.extract()?;
        let mut activation_memory: u64 = ob
            .getattr("activation_memory")
            .ok()
            .and_then(|value| value.extract().ok())
            .unwrap_or(0);
        let persistent_memory: u64 = ob
            .getattr("persistent_memory")
            .ok()
            .and_then(|value| value.extract().ok())
            .unwrap_or(0);
        if activation_memory == 0 && persistent_memory == 0 {
            activation_memory = mem_required;
        }
        Ok(Self {
            layer_index: ob.getattr("layer_index")?.extract()?,
            layer_name: ob.getattr("layer_name")?.extract()?,
            forward: ob.getattr("forward")?.extract()?,
            backward: ob.getattr("backward")?.extract()?,
            mem_required,
            activation_memory,
            persistent_memory,
        })
    }
}

pub struct StageExecutionResult {
    pub layers: (u32, u32),
    forward: f64,
    backward: f64,
    mem_required: u64,
    activation_memory: u64,
    persistent_memory: u64,
}

impl StageExecutionResult {
    pub fn new(layers: &[LayerExecutionResult]) -> Self {
        Self {
            layers: (layers[0].layer_index, layers[layers.len() - 1].layer_index + 1),
            forward: layers.iter().map(|layer| layer.forward).sum(),
            backward: layers.iter().map(|layer| layer.backward).sum(),
            mem_required: layers.iter().map(|layer| layer.mem_required).sum(),
            activation_memory: layers.iter().map(|layer| layer.activation_memory).sum(),
            persistent_memory: layers.iter().map(|layer| layer.persistent_memory).sum(),
        }
    }

    pub fn latency(&self) -> f64 {
        self.forward + self.backward
    }
}

#[derive(Clone)]
pub struct PipelineExecutionResult {
    pub stages: Vec<Arc<StageExecutionResult>>,
    pub t1: f64,
    pub t2: f64,
    pub t3: f64,
    pub kstar: usize,
}

impl PipelineExecutionResult {
    pub fn new(left: &Self, right: &Self) -> Self {
        let mut stages = left.stages.clone();
        stages.extend(right.stages.clone());
        let t1 = left.t1 + right.t1;
        let kstar = if left.stages[left.kstar].latency() > right.stages[right.kstar].latency() {
            left.kstar
        } else {
            left.stages.len() + right.kstar
        };
        let num_microbatches = 4 * stages.len();
        let t2 = (num_microbatches - stages.len() + kstar - 1) as f64 * stages[kstar].latency();
        let t3 = if kstar == left.kstar { left.t3 + right.t1 } else { right.t3 };
        Self {
            stages,
            t1,
            t2,
            t3,
            kstar,
        }
    }

    pub fn make_base_result(stage: Arc<StageExecutionResult>) -> Self {
        let latency = stage.latency();
        Self {
            stages: vec![stage],
            t1: latency,
            t2: 2.0 * latency,
            t3: latency,
            kstar: 0,
        }
    }

    pub fn latency_with_mb(&self, mb: u32) -> f64 {
        self.t1 + self.t2 + self.t3
            + ((mb as i32 - 4 * self.stages.len() as i32) as f64)
                * self.stages[self.kstar].latency()
    }
    pub fn latency(&self) -> f64 {
        self.t1 + self.t2 + self.t3
    }
    pub fn forward_time(&self) -> f64 {
        self.stages.iter().map(|stage| stage.forward).fold(0.0, f64::max)
    }
    pub fn backward_time(&self) -> f64 {
        self.stages.iter().map(|stage| stage.backward).fold(0.0, f64::max)
    }
    pub fn mem_required(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| stage.mem_required)
            .max()
            .unwrap_or(0)
    }
    pub fn activation_memory(&self) -> u64 {
        self.stages.iter().map(|stage| stage.activation_memory).max().unwrap_or(0)
    }
    pub fn persistent_memory(&self) -> u64 {
        self.stages.iter().map(|stage| stage.persistent_memory).max().unwrap_or(0)
    }
}

impl PartialEq for PipelineExecutionResult {
    fn eq(&self, other: &Self) -> bool {
        self.latency_with_mb(128) == other.latency_with_mb(128)
            && self.mem_required() == other.mem_required()
    }
}
impl Eq for PipelineExecutionResult {}
impl Ord for PipelineExecutionResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.latency_with_mb(128)
            .partial_cmp(&other.latency_with_mb(128))
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.mem_required().cmp(&other.mem_required()))
    }
}
impl PartialOrd for PipelineExecutionResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
