use crate::execution_result::*;
use dashmap::DashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::result::Result;

#[pyclass]
pub struct PipelineTemplateGenerator {
    layer_execution_results: Vec<LayerExecutionResult>,
    stage_execution_results: Vec<StageExecutionResult>,
    // Key: (num_stages, layer_start_index, layer_end_index)
    execution_result_cache: DashMap<(u32, usize, usize), Result<PipelineExecutionResult, String>>,
}

impl PipelineTemplateGenerator {
    pub fn new(model_name: &str, tag: &str, microbatch_size: u32) -> Self {
        PipelineTemplateGenerator {
            layer_execution_results: get_profile_results(model_name, tag),
            stage_execution_results: Vec::new(),
            execution_result_cache: DashMap::new(),
        }
    }

    fn divide_and_conquer(&mut self, num_nodes: u32) -> Result<(), String> {
        if num_nodes > self.layer_execution_results.len() as u32 {
            return Err("Invalid number of nodes".to_string());
        }

        let num_layers = self.layer_execution_results.len();

        // Put all base cases in the cache
        (0..num_layers).into_par_iter().for_each(|i| {
            (i..num_layers).into_par_iter().for_each(|j| {
                let key = (1, i, j);
                let stage_execution_result =
                    StageExecutionResult::new(&self.layer_execution_results[i..j], num_nodes);
                let pipeline_execution_result =
                    PipelineExecutionResult::make_base_result(stage_execution_result);
                self.execution_result_cache
                    .insert(key, Ok(pipeline_execution_result));
            });
        });

        // Compute the rest of the results, gradually increasing the number of stages
        // Number of stages can increase from 2 up to the number of nodes
        // (currently more than two stages cannot be assigned to a node)
        // Each number of stages all computations should be done before moving on to the next number of stages
        for num_stages in 2..num_nodes as u32 {
            rayon::scope(|s| {
                for i in 0..num_layers {
                    for j in i..num_layers {
                        let key = (num_stages, i, j);

                        // If number of layers is less than number of stages, skip it
                        // Cannot create specified number of stages with the given number of layers
                        if j - i + 1 < num_stages as usize {
                            self.execution_result_cache
                                .insert(key, Err("Infeasible case".to_string()));
                            continue;
                        }

                        // Spawn a task to compute the result for this subproblem.
                        s.spawn(move |_| {
                            // Iterate number of stages and layers to subproblems to find the best result
                            let best_result = (i..j)
                                .into_par_iter()
                                .map(|num_layers_left| {
                                    let mut result: Result<PipelineExecutionResult, String> =
                                        Err("Error in subproblem".to_string());

                                    for num_stages_left in 1..num_stages {
                                        let num_stages_right = num_stages - num_stages_left;

                                        // As we gradually increase the number of stages from 1,
                                        // we must have already computed the results for the subproblems
                                        let left = self
                                            .execution_result_cache
                                            .get(&(num_stages_left, i, num_layers_left))
                                            .unwrap();
                                        let right = self
                                            .execution_result_cache
                                            .get(&(num_stages_right, num_layers_left + 1, j))
                                            .unwrap();

                                        if left.is_err() || right.is_err() {
                                            continue;
                                        }

                                        // Merge two subproblems into a bigger PipelineExecutionResult
                                        let local_result = PipelineExecutionResult::new(
                                            left.unwrap(),
                                            right.unwrap(),
                                            num_stages,
                                        );
                                        if result.is_err()
                                            || local_result.cmp(&result.unwrap()) == Ordering::Less
                                        {
                                            result = Ok(local_result);
                                        }
                                    }

                                    result
                                })
                                .reduce(
                                    || Err("Error in subproblem".to_string()),
                                    |acc, result| match (result.is_err(), acc.is_err()) {
                                        (true, true) => return acc,
                                        (true, false) => return result,
                                        (false, true) => return acc,
                                        (false, false) => {
                                            if result.unwrap().cmp(&acc.unwrap()) == Ordering::Less
                                            {
                                                return result;
                                            } else {
                                                return acc;
                                            }
                                        }
                                    },
                                );

                            self.execution_result_cache.insert(key, best_result);
                        });
                    }
                }
            });
        }
        Ok(())
    }

    pub fn create_pipeline_template(&mut self, num_nodes: u32) {}
}

mod test {
    use super::*;
}
