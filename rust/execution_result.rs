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
    /// Section 4.1.2, Equation 1: one forward and backward pass for every stage.
    pub t1: f64,
    /// Section 4.1.2, Equation 3: work from the bottleneck through the last stage.
    pub t3: f64,
    /// Zero-based index of the rightmost stage with maximum forward + backward time.
    pub kstar: usize,
}

impl PipelineExecutionResult {
    /// Section 4.1.2, Equation 4 (conquer): evaluate one stage.
    pub fn from_stage(stage: StageExecutionResult) -> Self {
        let latency = stage.latency();
        Self {
            stages: vec![stage],
            t1: latency,
            t3: latency,
            kstar: 0,
        }
    }

    /// Section 4.1.2, Equations 1 and 3 (combine two subproblems).
    ///
    /// T2 is evaluated from the combined rightmost bottleneck by
    /// `iteration_time`, because its coefficient depends on the concrete Nb.
    pub fn combine(left: &Self, right: &Self) -> Self {
        let mut stages = left.stages.clone();
        stages.extend(right.stages.clone());
        let right_wins = right
            .bottleneck_latency()
            .total_cmp(&left.bottleneck_latency())
            != std::cmp::Ordering::Less;
        let (kstar, t3) = if right_wins {
            (left.stages.len() + right.kstar, right.t3)
        } else {
            (left.kstar, left.t3 + right.t1)
        };
        Self {
            stages,
            t1: left.t1 + right.t1,
            t3,
            kstar,
        }
    }

    pub fn bottleneck_stage(&self) -> &StageExecutionResult {
        &self.stages[self.kstar]
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

    /// Evaluate Section 4.1.2's T1 + T2 + T3 model.
    pub fn iteration_time(&self, num_microbatches: u32) -> f64 {
        // Equation 2: T2 = (Nb - S + k* - 1) * (F_k* + B_k*).
        let t2_coefficient = num_microbatches as i64 - self.stages.len() as i64
            + self.kstar as i64
            - 1;
        self.t1 + t2_coefficient as f64 * self.bottleneck_latency() + self.t3
    }

    /// The paper selects a template using the temporary planning value Nb = 4S.
    pub fn planning_iteration_time(&self) -> f64 {
        self.iteration_time((4 * self.stages.len()) as u32)
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
