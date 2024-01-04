use serde::{Deserialize, Serialize};
use std::clone::Clone;
use std::cmp::{Ordering, PartialEq};
use std::sync::Arc;

use csv;
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
pub struct LayerExecutionResult {
    layer_index: u32,
    layer_name: String,
    forward: f64,
    backward: f64,
    mem_required: u64,
}

impl LayerExecutionResult {
    pub fn new(
        layer_index: u32,
        layer_name: String,
        forward: f64,
        backward: f64,
        mem_required: u64,
    ) -> Self {
        LayerExecutionResult {
            layer_index,
            layer_name,
            forward,
            backward,
            mem_required,
        }
    }

    pub fn get_profile_results(
        tag: &str,
        oobleck_base_dir: Option<PathBuf>,
    ) -> Result<Vec<LayerExecutionResult>, std::io::Error> {
        let path = match oobleck_base_dir {
            Some(base_dir) => base_dir.join("profiles"),
            None => PathBuf::from("/tmp/oobleck/profiles/".to_string()),
        }
        .join(tag.to_string() + ".csv");

        let mut reader = csv::Reader::from_path(path)?;

        let mut data: Vec<LayerExecutionResult> = Vec::new();
        for result in reader.deserialize() {
            let record: LayerExecutionResult = result?;
            data.push(record);
        }
        drop(reader);

        Ok(data)
    }
}

pub struct StageExecutionResult {
    pub layers: (u32, u32),
    forward: f64,
    backward: f64,
    mem_required: u64,
}

impl StageExecutionResult {
    pub fn new(layers: &[LayerExecutionResult]) -> Self {
        let mut forward = 0.0;
        let mut backward = 0.0;
        let mut mem_required = 0;

        for layer in layers {
            forward += layer.forward;
            backward += layer.backward;
            mem_required += layer.mem_required;
        }

        StageExecutionResult {
            layers: (
                layers[0].layer_index,
                layers[layers.len() - 1].layer_index + 1,
            ),
            forward,
            backward,
            mem_required,
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
    pub fn new(left: &PipelineExecutionResult, right: &PipelineExecutionResult) -> Self {
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

        let t3 = if kstar == left.kstar {
            left.t3 + right.t1
        } else {
            right.t3
        };

        PipelineExecutionResult {
            stages,
            t1,
            t2,
            t3,
            kstar,
        }
    }
    pub fn make_base_result(stage: Arc<StageExecutionResult>) -> Self {
        let latency = stage.latency();
        PipelineExecutionResult {
            stages: vec![stage],
            t1: latency,
            t2: 2.0 * (latency),
            t3: latency,
            kstar: 0,
        }
    }
    pub fn latency(&self) -> f64 {
        self.t1 + self.t2 + self.t3
    }
    pub fn mem_required(&self) -> u64 {
        self.stages.iter().fold(0, |acc, x| acc + x.mem_required)
    }
    pub fn get_modules_per_stage(&self, layers: &Vec<LayerExecutionResult>) -> Vec<Vec<String>> {
        let mut modules_per_stage: Vec<Vec<String>> = Vec::new();
        for stage in &self.stages {
            let mut modules: Vec<String> = Vec::new();
            for layer in &layers[stage.layers.0 as usize..stage.layers.1 as usize] {
                modules.push(layer.layer_name.clone());
            }
            modules_per_stage.push(modules);
        }
        modules_per_stage
    }
}

impl PartialEq for PipelineExecutionResult {
    fn eq(&self, other: &Self) -> bool {
        self.latency() == other.latency() && self.mem_required() == other.mem_required()
    }
}

impl Eq for PipelineExecutionResult {}

impl Ord for PipelineExecutionResult {
    fn cmp(&self, other: &Self) -> Ordering {
        if self == other {
            Ordering::Equal
        } else {
            if self.latency() < other.latency() {
                Ordering::Less
            } else if self.latency() > other.latency() {
                Ordering::Greater
            } else {
                if self.mem_required() < other.mem_required() {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
        }
    }
}

impl PartialOrd for PipelineExecutionResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
