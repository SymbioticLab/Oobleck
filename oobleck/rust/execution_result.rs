use std::cmp::{Ordering, PartialEq};

pub struct LayerExecutionResult {
    layer_index: u32,
    forward: f64,
    backward: f64,
    mem_required: u64,
}

pub struct StageExecutionResult {
    layers: (u32, u32),
    forward: f64,
    backward: f64,
    mem_required: u64,
}

impl StageExecutionResult {
    pub fn new(layers: &[LayerExecutionResult], num_nodes: u32) -> Self {
        let mut forward = 0.0;
        let mut backward = 0.0;
        let mut mem_required = 0;

        for layer in layers {
            forward += layer.forward;
            backward += layer.backward;
            mem_required += layer.mem_required;
        }

        StageExecutionResult {
            layers: (layers[0].layer_index, layers[layers.len() - 1].layer_index),
            forward: forward,
            backward: backward,
            mem_required: mem_required,
        }
    }
}

#[derive(PartialEq, Eq)]
pub struct PipelineExecutionResult {
    stages: Vec<StageExecutionResult>,
    t1: f64,
    t2: f64,
    t3: f64,
    kstar: i32,
}

impl PipelineExecutionResult {
    pub fn new(
        left: PipelineExecutionResult,
        right: PipelineExecutionResult,
        num_nodes: u32,
    ) -> Self {
    }
    pub fn make_base_result(stage: StageExecutionResult) -> Self {
        PipelineExecutionResult {
            stages: vec![stage],
            t1: 0.0,
            t2: 0.0,
            t3: 0.0,
            kstar: 0,
        }
    }
    pub fn latency(&self) -> f64 {
        self.t1 + self.t2 + self.t3
    }
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }
    pub fn mem_required(&self) -> u64 {
        self.stages.iter().fold(0, |acc, x| acc + x.mem_required)
    }
    pub fn layers(&self) -> (u32, u32) {
        (
            self.stages[0].layers.0,
            self.stages[self.stages.len() - 1].layers.1,
        )
    }
}

impl Ord for PipelineExecutionResult {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.latency() < other.latency() {
            Ordering::Less
        } else if self.latency() > other.latency() {
            Ordering::Greater
        } else {
            if self.mem_required() < other.mem_required() {
                Ordering::Less
            } else if self.mem_required() > other.mem_required() {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    }
}

impl PartialOrd for PipelineExecutionResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
