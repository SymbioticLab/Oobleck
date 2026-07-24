use crate::PlannerError;
use pyo3::conversion::FromPyObject;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[derive(Clone, Debug)]
pub struct LayerExecutionResult {
    pub layer_index: u32,
    pub layer_name: String,
    pub forward: f64,
    pub backward: f64,
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
            activation_memory,
            persistent_memory,
        })
    }
}

impl LayerExecutionResult {
    pub fn validate(&self, expected_index: usize) -> Result<(), PlannerError> {
        if self.layer_index as usize != expected_index {
            return Err(PlannerError::invalid_input(format!(
                "profile_data must have contiguous layer indices; expected {expected_index}, got {}",
                self.layer_index
            )));
        }
        if self.layer_name.is_empty() {
            return Err(PlannerError::invalid_input(format!(
                "layer {expected_index} must have a non-empty name"
            )));
        }
        if !self.forward.is_finite()
            || !self.backward.is_finite()
            || !(self.forward + self.backward).is_finite()
            || self.forward < 0.0
            || self.backward < 0.0
        {
            return Err(PlannerError::invalid_input(format!(
                "layer {expected_index} timings must be finite and non-negative"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StageExecutionResult {
    pub layers: (u32, u32),
    pub forward: f64,
    pub backward: f64,
    pub activation_memory: u64,
    pub persistent_memory: u64,
}

impl StageExecutionResult {
    pub fn latency(&self) -> f64 {
        self.forward + self.backward
    }

    pub fn fits_one_microbatch(&self, device_memory_bytes: Option<u64>) -> bool {
        device_memory_bytes.map_or(true, |device_memory| {
            self.persistent_memory
                .checked_add(self.activation_memory)
                .is_some_and(|required| required <= device_memory)
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PipelineExecutionResult {
    pub stages: Vec<StageExecutionResult>,
}

impl PipelineExecutionResult {
    pub fn bottleneck_stage(&self) -> &StageExecutionResult {
        self.stages
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.latency()
                    .total_cmp(&right.latency())
                    .then_with(|| left_index.cmp(right_index))
            })
            .map(|(_, stage)| stage)
            .expect("a planned pipeline always has at least one stage")
    }

    pub fn bottleneck_latency(&self) -> f64 {
        self.bottleneck_stage().latency()
    }

    pub fn forward_time(&self) -> f64 {
        self.bottleneck_stage().forward
    }

    pub fn backward_time(&self) -> f64 {
        self.bottleneck_stage().backward
    }

    pub fn activation_memory(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| stage.activation_memory)
            .max()
            .unwrap_or(0)
    }

    pub fn persistent_memory(&self) -> u64 {
        self.stages
            .iter()
            .map(|stage| stage.persistent_memory)
            .max()
            .unwrap_or(0)
    }

    pub fn max_microbatches(&self, device_memory_bytes: Option<u64>) -> Option<u64> {
        let device_memory = device_memory_bytes?;
        self.stages
            .iter()
            .filter(|stage| stage.activation_memory > 0)
            .map(|stage| {
                device_memory
                    .checked_sub(stage.persistent_memory)
                    .map(|available| available / stage.activation_memory)
                    .unwrap_or(0)
            })
            .min()
    }
}
