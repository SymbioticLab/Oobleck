use crate::execution_result::*;
use dashmap::DashMap;
use env_logger;
use log;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::result::Result;
use std::sync::Arc;

pub struct PipelineTemplateGenerator {
    layer_execution_results: Vec<LayerExecutionResult>,
    // Key: (layer_start_index, layer_end_index)
    stage_execution_results: DashMap<(usize, usize), Arc<StageExecutionResult>>,
    // Key: (num_stages, layer_start_index, layer_end_index)
    execution_result_cache: DashMap<(u32, usize, usize), Result<PipelineExecutionResult, String>>,
}

impl PipelineTemplateGenerator {
    pub fn new(model_name: &str, tag: &str) -> Self {
        PipelineTemplateGenerator {
            layer_execution_results: LayerExecutionResult::get_profile_results(model_name, tag),
            stage_execution_results: DashMap::new(),
            execution_result_cache: DashMap::new(),
        }
    }

    pub fn divide_and_conquer(&mut self, max_num_nodes: u32) -> Result<(), String> {
        if !self.stage_execution_results.is_empty() {
            return Ok(());
        }

        let num_layers = self.layer_execution_results.len();

        if max_num_nodes as usize > num_layers {
            return Err("Invalid number of nodes".to_string());
        }

        // Put all base cases in the cache
        (0..num_layers).into_par_iter().for_each(|i| {
            ((i + 1)..=num_layers).into_par_iter().for_each(|j| {
                let stage_execution_result = Arc::new(StageExecutionResult::new(
                    &self.layer_execution_results[i..j],
                ));
                log::debug!(
                    "StageExecutionResult({}, {})  -> {}",
                    stage_execution_result.layers.0,
                    stage_execution_result.layers.1,
                    stage_execution_result.latency()
                );
                self.stage_execution_results
                    .insert((i, j), stage_execution_result.clone());

                let pipeline_execution_result =
                    PipelineExecutionResult::make_base_result(stage_execution_result);
                log::debug!(
                    "PipelineExecutionResult({}, {}, {}) -> {}",
                    1,
                    i,
                    j,
                    pipeline_execution_result.latency()
                );
                self.execution_result_cache
                    .insert((1, i, j), Ok(pipeline_execution_result));
            });
        });

        log::debug!("Base cases inserted into the cache");

        // Compute the rest of the results, gradually increasing the number of stages
        // Number of stages can increase from 2 up to the number of nodes
        // (currently more than two stages cannot be assigned to a node)
        // Each number of stages all computations should be done before moving on to the next number of stages
        for num_stages in 2..=max_num_nodes as u32 {
            (0..num_layers).into_par_iter().for_each(|i| {
                ((i + 1)..=num_layers).into_par_iter().for_each(|j| {
                    let key = (num_stages, i, j);

                    // If number of layers is less than number of stages, skip it
                    // Cannot create specified number of stages with the given number of layers
                    if j - i < num_stages as usize {
                        self.execution_result_cache
                            .insert(key, Err("Infeasible case".to_string()));
                        return;
                    }

                    // Spawn a task to compute the result for this subproblem.
                    let best_result = (i..j)
                        .into_par_iter()
                        .map(|num_layers_left| {
                            let mut result: Result<PipelineExecutionResult, String> =
                                Err("Error in subproblem".to_string());

                            for num_stages_left in 1..num_stages {
                                let num_stages_right = num_stages - num_stages_left;

                                if num_layers_left - i == 0 || j - num_layers_left == 0 {
                                    continue;
                                }

                                // As we gradually increase the number of stages from 1,
                                // we must have already computed the results for the subproblems
                                let left = self
                                    .execution_result_cache
                                    .get(&(num_stages_left, i, num_layers_left))
                                    .unwrap();
                                let right = self
                                    .execution_result_cache
                                    .get(&(num_stages_right, num_layers_left, j))
                                    .unwrap();

                                log::debug!(
                                    "({}, {}, {}) Getting subproblems: ({}, {}, {}) -> {} + ({}, {}, {}) -> {}",
                                    num_stages,
                                    i,
                                    j,
                                    num_stages_left,
                                    i,
                                    num_layers_left,
                                    if left.is_ok() {
                                        left.as_ref().unwrap().latency()
                                    } else {
                                        0.0
                                    },
                                    num_stages_right,
                                    num_layers_left,
                                    j,
                                    if right.is_ok() {
                                        right.as_ref().unwrap().latency()
                                    } else {
                                        0.0
                                    }
                                );

                                if left.is_err() || right.is_err() {
                                    continue;
                                }

                                // Merge two subproblems into a bigger PipelineExecutionResult
                                let local_result = PipelineExecutionResult::new(
                                    left.as_ref().unwrap(),
                                    right.as_ref().unwrap(),
                                );
                                if result.is_err()
                                    || local_result.cmp(result.as_ref().unwrap()) == Ordering::Less
                                {
                                    result = Ok(local_result);
                                }
                            }

                            result
                        })
                        .reduce(
                            || Err("Error in subproblem".to_string()),
                            |acc, result| {
                                if result.is_err() {
                                    return acc;
                                }
                                else if acc.is_err() {
                                    return result;
                                }
                                else if result.as_ref().unwrap() < acc.as_ref().unwrap() {
                                    return result;
                                } else {
                                    return acc;
                                }
                            },
                        );

                    log::debug!(
                        "PipelineExecutionResult({}, {}, {}) -> {}",
                        num_stages,
                        i,
                        j,
                        if best_result.is_ok() {
                            best_result.as_ref().unwrap().latency()
                        } else {
                            0.0
                        }
                    );
                    self.execution_result_cache.insert(key, best_result);
                })
            });
        }
        Ok(())
    }

    pub fn get_pipeline_template(&self, num_nodes: u32) -> Vec<Vec<String>> {
        log::debug!(
            "get_pipeline_template({}, {}, {})",
            num_nodes,
            0,
            self.layer_execution_results.len()
        );
        self.execution_result_cache
            .get(&(num_nodes, 0, self.layer_execution_results.len()))
            .unwrap()
            .as_ref()
            .unwrap()
            .get_modules_per_stage(&self.layer_execution_results)
    }
}

#[pyfunction]
pub fn create_pipeline_templates(
    model_name: &str,
    tag: &str,
    mut nodes: Vec<u32>,
) -> HashMap<u32, Vec<Vec<String>>> {
    env_logger::init();
    nodes.sort();

    let mut generator = PipelineTemplateGenerator::new(model_name, tag);
    let _ = generator.divide_and_conquer(nodes[nodes.len() - 1]);

    let mut results: HashMap<u32, Vec<Vec<String>>> = HashMap::new();
    for num_node in nodes {
        results.insert(num_node, generator.get_pipeline_template(num_node));
    }

    results
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn prepare_profile_file() {
        let model_name = "gpt2";
        let tag = "test";
        let path =
            PathBuf::from("/tmp/oobleck/profiles/".to_string() + model_name + "__" + tag + ".csv");
        fs::create_dir_all(path.parent().unwrap()).unwrap();

        let mut writer = csv::Writer::from_path(path).unwrap();
        for i in 0..4 {
            writer
                .serialize(LayerExecutionResult::new(
                    i,
                    format!("layer{}", i),
                    (i + 1) as f64 * 2.0,
                    (i + 1) as f64 * 2.0,
                    (i + 1) as u64,
                ))
                .unwrap();
        }
    }

    #[test]
    fn test_divide_and_conquer_base_only() {
        prepare_profile_file();
        let template = create_pipeline_templates("gpt2", "test", vec![1]);
        assert_eq!(template.len(), 1);
        assert_eq!(template[&1].len(), 1);
        assert_eq!(
            template[&1][0],
            vec!["layer0", "layer1", "layer2", "layer3"]
        );
    }

    #[test]
    fn test_divide_and_conquer_divide() {
        prepare_profile_file();
        let templates = create_pipeline_templates("gpt2", "test", vec![1, 2]);
        assert_eq!(templates.len(), 2);
        assert_eq!(
            templates[&1][0],
            vec!["layer0", "layer1", "layer2", "layer3"]
        );
        assert_eq!(templates[&2][0], vec!["layer0", "layer1", "layer2"]);
        assert_eq!(templates[&2][1], vec!["layer3"]);
    }
}
