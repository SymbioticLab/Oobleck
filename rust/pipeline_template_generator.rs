use crate::execution_result::*;
use crate::PlannerError;
use dashmap::DashMap;
use log;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::path::PathBuf;
use std::result::Result;
use std::sync::Arc;

pub struct PipelineTemplateGenerator {
    pub layer_execution_results: Vec<LayerExecutionResult>,
    // Key: (layer_start_index, layer_end_index)
    stage_execution_results: DashMap<(usize, usize), Arc<StageExecutionResult>>,
    // Key: (num_stages, layer_start_index, layer_end_index)
    execution_result_cache: DashMap<(u32, usize, usize), Result<PipelineExecutionResult, String>>,
}

impl PipelineTemplateGenerator {
    pub fn new(tag: &str, oobleck_base_dir: Option<PathBuf>) -> Self {
        PipelineTemplateGenerator {
            layer_execution_results: LayerExecutionResult::get_profile_results(
                tag,
                oobleck_base_dir,
            )
            .unwrap(),
            stage_execution_results: DashMap::new(),
            execution_result_cache: DashMap::new(),
        }
    }

    pub fn divide_and_conquer(&mut self, max_num_nodes: u32) -> Result<(), PlannerError> {
        if !self.stage_execution_results.is_empty() {
            return Ok(());
        }

        let num_layers = self.layer_execution_results.len();

        if max_num_nodes as usize > num_layers {
            return Err(PlannerError::new("Invalid number of nodes"));
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
                                } else if acc.is_err() {
                                    return result;
                                } else if result.as_ref().unwrap() < acc.as_ref().unwrap() {
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

    pub fn get_pipeline_template(
        &self,
        num_nodes: u32,
    ) -> Result<PipelineExecutionResult, PlannerError> {
        log::debug!(
            "get_pipeline_template({}, {}, {})",
            num_nodes,
            0,
            self.layer_execution_results.len()
        );

        let result =
            self.execution_result_cache
                .get(&(num_nodes, 0, self.layer_execution_results.len()));

        match result {
            Some(result) => Ok(result.value().clone().unwrap()),
            None => Err(PlannerError::new(
                format!("No pipeline template for {} nodes", num_nodes).as_str(),
            ))?,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn prepare(
        num_layers: u32,
        same_latency: bool,
        mut num_nodes: Vec<u32>,
    ) -> Result<PipelineTemplateGenerator, PlannerError> {
        let tag = "gpt2-test";
        let base_dir = TempDir::new().unwrap().path().to_path_buf();
        let path = base_dir.join("profiles").join(tag.to_string() + ".csv");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let _ = fs::remove_file(&path);

        let mut writer = csv::Writer::from_path(path).unwrap();
        for i in 0..num_layers {
            writer
                .serialize(LayerExecutionResult::new(
                    i,
                    format!("layer{}", i),
                    if same_latency {
                        1 as f64
                    } else {
                        (i + 1) as f64
                    },
                    if same_latency {
                        1 as f64
                    } else {
                        (i + 1) as f64
                    },
                    if same_latency {
                        1 as u64
                    } else {
                        (i + 1) as u64
                    },
                ))
                .unwrap();
        }
        writer.flush().unwrap();
        drop(writer);

        num_nodes.sort();

        let mut generator = PipelineTemplateGenerator::new(tag, Some(base_dir));
        generator.divide_and_conquer(num_nodes[num_nodes.len() - 1])?;
        Ok(generator)
    }

    #[test]
    fn test_return_no_template_for_too_large_num_nodes() {
        let generator = prepare(6, true, vec![7]);
        assert!(generator.is_err());

        let generator = prepare(6, true, vec![6]);
        assert!(generator.is_ok());
        assert!(generator.unwrap().get_pipeline_template(7).is_err());
    }

    #[test]
    fn test_all_layers_covered() {
        let generator = prepare(6, false, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let expected_layers: Vec<u32> = (0..6).map(|i| i).collect();

        for i in 1..=6 {
            let template = generator.get_pipeline_template(i).unwrap();
            let mut covered_layers: Vec<u32> = Vec::new();
            for stage in template.stages.iter() {
                for layer in stage.layers.0..stage.layers.1 {
                    covered_layers.push(layer);
                }
            }
            assert_eq!(covered_layers, expected_layers);
        }
    }

    #[test]
    fn test_divide_and_conquer_base_only() {
        let generator = prepare(6, false, vec![1]).unwrap();
        let template = generator.get_pipeline_template(1).unwrap();

        assert_eq!(template.stages.len(), 1);
        assert_eq!(template.stages[0].layers, (0, 6));
    }

    #[test]
    fn test_divide_and_conquer_divide() {
        // Uneven distribution test
        let generator = prepare(6, false, vec![1, 2]).unwrap();

        // Template for 1 node
        let template = generator.get_pipeline_template(1).unwrap();
        assert_eq!(template.stages.len(), 1);
        assert_eq!(template.stages[0].layers, (0, 6));

        let template = generator.get_pipeline_template(2).unwrap();
        assert_eq!(template.stages.len(), 2);
        assert_eq!(template.stages[0].layers, (0, 4));
        assert_eq!(template.stages[1].layers, (4, 6));

        let generator = prepare(6, true, vec![1, 2]).unwrap();
        let template = generator.get_pipeline_template(2).unwrap();
        assert_eq!(template.stages.len(), 2);
        assert_eq!(template.stages[0].layers, (0, 3));
        assert_eq!(template.stages[1].layers, (3, 6));
    }

    #[test]
    fn test_divide_and_conquer_divide2() {
        let generator = prepare(6, false, vec![2, 3, 4]).unwrap();
        let template = generator.get_pipeline_template(2).unwrap();
        assert_eq!(template.stages.len(), 2);
        assert_eq!(template.stages[0].layers, (0, 4));
        assert_eq!(template.stages[1].layers, (4, 6));

        let template = generator.get_pipeline_template(3).unwrap();
        assert_eq!(template.stages.len(), 3);
        assert_eq!(template.stages[0].layers, (0, 3));
        assert_eq!(template.stages[1].layers, (3, 5));
        assert_eq!(template.stages[2].layers, (5, 6));

        let template = generator.get_pipeline_template(4).unwrap();
        assert_eq!(template.stages.len(), 4);
        assert_eq!(template.stages[0].layers, (0, 3));
        assert_eq!(template.stages[1].layers, (3, 4));
        assert_eq!(template.stages[2].layers, (4, 5));
        assert_eq!(template.stages[3].layers, (5, 6));
    }

    #[test]
    fn test_measure_time_of_large_model() {
        let generator = prepare(96, false, vec![64]).unwrap();
        for i in 1..=64 {
            let template_result = generator.get_pipeline_template(i);
            assert!(template_result.is_ok());
            assert_eq!(template_result.unwrap().stages.len(), i as usize);
        }
    }
}
