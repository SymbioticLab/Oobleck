use csv;
use pyo3::prelude::*;
use serde::Deserialize;
use std::cmp::{Ordering, PartialEq};
use std::path::PathBuf;

#[pyfunction]
pub fn get_profile_results(model_name: &str, tag: &str) -> Vec<LayerExecutionResult> {
    let path =
        PathBuf::from("/tmp/oobleck/profiles/".to_string() + model_name + "__" + tag + ".csv");
    let mut reader = csv::Reader::from_path(path).unwrap();

    let mut data: Vec<LayerExecutionResult> = Vec::new();
    for result in reader.deserialize() {
        let record: LayerExecutionResult = result.unwrap();
        data.push(record);
    }

    data
}

#[pyclass]
#[derive(Deserialize)]
pub struct LayerExecutionResult {
    layer_index: u32,
    layer_name: String,
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

    pub fn latency(&self) -> f64 {
        self.forward + self.backward
    }
}

#[derive(PartialEq, Eq)]
pub struct PipelineExecutionResult {
    stages: Vec<StageExecutionResult>,
    t1: f64,
    t2: f64,
    t3: f64,
    kstar: usize,
}

impl PipelineExecutionResult {
    pub fn new(
        left: PipelineExecutionResult,
        right: PipelineExecutionResult,
        num_nodes: u32,
    ) -> Self {
        let t1 = left.t1 + right.t1;

        let kstar = if left.stages[left.kstar].latency() < right.stages[right.kstar].latency() {
            (left.kstar, left)
        } else {
            (left.stages.len() + right.kstar, right)
        };

        let num_stages = left.stages.len() + right.stages.len();
        let num_microbatches = 4 * num_stages;
        let t2 = (num_microbatches - num_stages + kstar.0 - 1) as f64
            * kstar.1.stages[kstar.0].latency();

        let t3 = if kstar.0 == left.kstar {
            left.t3 + right.t1
        } else {
            right.t3
        };

        PipelineExecutionResult {
            stages: left.stages.concat(right.stages),
            t1: t1,
            t2: t2,
            t3: t3,
            kstar: kstar.0,
        }
    }
    pub fn make_base_result(stage: StageExecutionResult) -> Self {
        PipelineExecutionResult {
            stages: vec![stage],
            t1: stage.latency(),
            t2: 2.0 * (stage.latency()),
            t3: stage.latency(),
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
