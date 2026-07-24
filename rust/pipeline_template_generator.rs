use crate::execution_result::{
    LayerExecutionResult, PipelineExecutionResult, StageExecutionResult,
};
use crate::PlannerError;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
struct PartitionState {
    bottleneck_latency: f64,
    stage_starts: Vec<usize>,
}

impl PartitionState {
    fn compare(&self, other: &Self) -> Ordering {
        self.bottleneck_latency
            .total_cmp(&other.bottleneck_latency)
            .then_with(|| self.stage_starts.cmp(&other.stage_starts))
    }
}

pub struct PipelineTemplateGenerator {
    layer_execution_results: Vec<LayerExecutionResult>,
    stage_execution_results: Vec<Vec<Option<StageExecutionResult>>>,
    pipeline_execution_results: Vec<Option<PipelineExecutionResult>>,
}

impl PipelineTemplateGenerator {
    pub fn new(profile_data: Vec<LayerExecutionResult>) -> Result<Self, PlannerError> {
        if profile_data.is_empty() {
            return Err(PlannerError::invalid_input(
                "profile_data must contain at least one layer",
            ));
        }
        for (index, layer) in profile_data.iter().enumerate() {
            layer.validate(index)?;
        }

        let stage_execution_results = Self::build_stage_results(&profile_data)?;
        Ok(Self {
            layer_execution_results: profile_data,
            stage_execution_results,
            pipeline_execution_results: Vec::new(),
        })
    }

    fn checked_prefix(
        profile_data: &[LayerExecutionResult],
        value: impl Fn(&LayerExecutionResult) -> u64,
        name: &str,
    ) -> Result<Vec<u64>, PlannerError> {
        let mut prefix = Vec::with_capacity(profile_data.len() + 1);
        prefix.push(0_u64);
        for layer in profile_data {
            let next = prefix
                .last()
                .expect("prefix contains its zero element")
                .checked_add(value(layer))
                .ok_or_else(|| {
                    PlannerError::invalid_input(format!(
                        "{name} overflows u64 while aggregating profile_data"
                    ))
                })?;
            prefix.push(next);
        }
        Ok(prefix)
    }

    fn build_stage_results(
        profile_data: &[LayerExecutionResult],
    ) -> Result<Vec<Vec<Option<StageExecutionResult>>>, PlannerError> {
        let num_layers = profile_data.len();
        let mut forward_prefix = Vec::with_capacity(num_layers + 1);
        let mut backward_prefix = Vec::with_capacity(num_layers + 1);
        forward_prefix.push(0.0);
        backward_prefix.push(0.0);
        for layer in profile_data {
            forward_prefix.push(
                forward_prefix
                    .last()
                    .expect("prefix contains its zero element")
                    + layer.forward,
            );
            backward_prefix.push(
                backward_prefix
                    .last()
                    .expect("prefix contains its zero element")
                    + layer.backward,
            );
        }
        let activation_prefix = Self::checked_prefix(
            profile_data,
            |layer| layer.activation_memory,
            "activation_memory",
        )?;
        let persistent_prefix = Self::checked_prefix(
            profile_data,
            |layer| layer.persistent_memory,
            "persistent_memory",
        )?;

        if !(forward_prefix[num_layers] + backward_prefix[num_layers]).is_finite() {
            return Err(PlannerError::invalid_input(
                "aggregate profile timing overflows f64",
            ));
        }

        let mut results = vec![vec![None; num_layers + 1]; num_layers];
        for start in 0..num_layers {
            for end in (start + 1)..=num_layers {
                results[start][end] = Some(StageExecutionResult {
                    layers: (start as u32, end as u32),
                    forward: forward_prefix[end] - forward_prefix[start],
                    backward: backward_prefix[end] - backward_prefix[start],
                    activation_memory: activation_prefix[end] - activation_prefix[start],
                    persistent_memory: persistent_prefix[end] - persistent_prefix[start],
                });
            }
        }
        Ok(results)
    }

    fn stage(&self, start: usize, end: usize) -> &StageExecutionResult {
        self.stage_execution_results[start][end]
            .as_ref()
            .expect("only non-empty contiguous stages are requested")
    }

    pub fn plan(
        &mut self,
        max_resource_count: u32,
        device_memory_bytes: Option<u64>,
    ) -> Result<(), PlannerError> {
        let num_layers = self.layer_execution_results.len();
        let max_stages = max_resource_count as usize;
        if max_stages == 0 {
            return Err(PlannerError::invalid_input(
                "resource counts must be positive",
            ));
        }
        if max_stages > num_layers {
            return Err(PlannerError::invalid_input(format!(
                "resource count {max_stages} exceeds the number of profiled layers {num_layers}"
            )));
        }

        let mut states = vec![vec![None::<PartitionState>; num_layers + 1]; max_stages + 1];
        states[0][0] = Some(PartitionState {
            bottleneck_latency: 0.0,
            stage_starts: Vec::new(),
        });

        for num_stages in 1..=max_stages {
            for end in num_stages..=num_layers {
                let mut best: Option<PartitionState> = None;
                for start in (num_stages - 1)..end {
                    let Some(previous) = states[num_stages - 1][start].as_ref() else {
                        continue;
                    };
                    let stage = self.stage(start, end);
                    if !stage.fits_one_microbatch(device_memory_bytes) {
                        continue;
                    }
                    let mut stage_starts = previous.stage_starts.clone();
                    stage_starts.push(start);
                    let candidate = PartitionState {
                        bottleneck_latency: previous.bottleneck_latency.max(stage.latency()),
                        stage_starts,
                    };
                    if best
                        .as_ref()
                        .map_or(true, |current| candidate.compare(current) == Ordering::Less)
                    {
                        best = Some(candidate);
                    }
                }
                states[num_stages][end] = best;
            }
        }

        self.pipeline_execution_results = vec![None; max_stages + 1];
        for num_stages in 1..=max_stages {
            let Some(state) = states[num_stages][num_layers].as_ref() else {
                continue;
            };
            let mut stages = Vec::with_capacity(num_stages);
            for (index, start) in state.stage_starts.iter().copied().enumerate() {
                let end = state
                    .stage_starts
                    .get(index + 1)
                    .copied()
                    .unwrap_or(num_layers);
                stages.push(self.stage(start, end).clone());
            }
            self.pipeline_execution_results[num_stages] =
                Some(PipelineExecutionResult { stages });
        }
        Ok(())
    }

    pub fn get_pipeline_template(
        &self,
        resource_count: u32,
    ) -> Result<PipelineExecutionResult, PlannerError> {
        self.pipeline_execution_results
            .get(resource_count as usize)
            .and_then(Option::as_ref)
            .cloned()
            .ok_or_else(|| {
                PlannerError::infeasible(format!(
                    "pipeline template for resource count {resource_count} cannot fit one microbatch in device memory"
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(index: u32, latency: f64, activation_memory: u64) -> LayerExecutionResult {
        LayerExecutionResult {
            layer_index: index,
            layer_name: format!("layer{index}"),
            forward: latency,
            backward: 0.0,
            activation_memory,
            persistent_memory: 0,
        }
    }

    fn prepare(
        latencies: &[f64],
        max_resource_count: u32,
        device_memory_bytes: Option<u64>,
    ) -> Result<PipelineTemplateGenerator, PlannerError> {
        let profile = latencies
            .iter()
            .enumerate()
            .map(|(index, latency)| layer(index as u32, *latency, 1))
            .collect();
        let mut generator = PipelineTemplateGenerator::new(profile)?;
        generator.plan(max_resource_count, device_memory_bytes)?;
        Ok(generator)
    }

    fn ranges(result: &PipelineExecutionResult) -> Vec<(u32, u32)> {
        result.stages.iter().map(|stage| stage.layers).collect()
    }

    fn brute_force_bottleneck(
        latencies: &[f64],
        stages: usize,
        start: usize,
    ) -> Option<f64> {
        if stages == 1 {
            return Some(latencies[start..].iter().sum());
        }
        let mut best: Option<f64> = None;
        for end in (start + 1)..=(latencies.len() - stages + 1) {
            let first: f64 = latencies[start..end].iter().sum();
            let suffix = brute_force_bottleneck(latencies, stages - 1, end)?;
            let candidate = first.max(suffix);
            best = Some(best.map_or(candidate, |current| current.min(candidate)));
        }
        best
    }

    #[test]
    fn rejects_invalid_profiles_and_resource_counts() {
        assert!(PipelineTemplateGenerator::new(Vec::new()).is_err());
        let mut invalid = layer(1, 1.0, 1);
        assert!(PipelineTemplateGenerator::new(vec![invalid.clone()]).is_err());
        invalid.layer_index = 0;
        invalid.forward = f64::NAN;
        assert!(PipelineTemplateGenerator::new(vec![invalid]).is_err());

        let mut generator = prepare(&[1.0, 2.0], 2, None).unwrap();
        assert!(generator.plan(0, None).is_err());
        assert!(generator.plan(3, None).is_err());
    }

    #[test]
    fn returns_contiguous_globally_optimal_partitions() {
        let generator = prepare(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4, None).unwrap();
        for stage_count in 1..=4 {
            let result = generator.get_pipeline_template(stage_count).unwrap();
            assert_eq!(result.stages.len(), stage_count as usize);
            assert_eq!(result.stages.first().unwrap().layers.0, 0);
            assert_eq!(result.stages.last().unwrap().layers.1, 6);
            for adjacent in result.stages.windows(2) {
                assert_eq!(adjacent[0].layers.1, adjacent[1].layers.0);
            }
            assert_eq!(
                result.bottleneck_latency(),
                brute_force_bottleneck(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], stage_count as usize, 0)
                    .unwrap()
            );
        }
        assert_eq!(
            ranges(&generator.get_pipeline_template(2).unwrap()),
            vec![(0, 4), (4, 6)]
        );
    }

    #[test]
    fn fixes_the_legacy_non_optimal_subproblem_counterexample() {
        let latencies = [91.0, 45.0, 29.0, 75.0, 73.0, 66.0, 99.0, 81.0, 83.0, 83.0];
        let generator = prepare(&latencies, 6, None).unwrap();
        let result = generator.get_pipeline_template(6).unwrap();
        assert_eq!(
            ranges(&result),
            vec![(0, 2), (2, 4), (4, 6), (6, 7), (7, 9), (9, 10)]
        );
        assert_eq!(
            result.bottleneck_latency(),
            brute_force_bottleneck(&latencies, 6, 0).unwrap()
        );
    }

    #[test]
    fn uses_lexicographically_smallest_cuts_for_equal_costs() {
        let generator = prepare(&[1.0, 1.0, 1.0], 2, None).unwrap();
        assert_eq!(
            ranges(&generator.get_pipeline_template(2).unwrap()),
            vec![(0, 1), (1, 3)]
        );
    }

    #[test]
    fn excludes_fast_partitions_that_do_not_fit_memory() {
        let profile = vec![layer(0, 1.0, 8), layer(1, 5.0, 5), layer(2, 6.0, 1)];
        let mut generator = PipelineTemplateGenerator::new(profile).unwrap();
        generator.plan(2, Some(10)).unwrap();
        let result = generator.get_pipeline_template(2).unwrap();
        assert_eq!(ranges(&result), vec![(0, 1), (1, 3)]);
        assert_eq!(result.bottleneck_latency(), 11.0);
        assert_eq!(result.max_microbatches(Some(10)), Some(1));
    }

    #[test]
    fn plans_realistic_layer_and_resource_counts() {
        let latencies: Vec<f64> = (1..=96).map(f64::from).collect();
        let generator = prepare(&latencies, 64, None).unwrap();
        for resource_count in 1..=64 {
            let result = generator.get_pipeline_template(resource_count).unwrap();
            assert_eq!(result.stages.len(), resource_count as usize);
        }
    }

    #[test]
    fn rejects_aggregate_timing_overflow() {
        let profile = vec![layer(0, 1.0e308, 1), layer(1, 1.0e308, 1)];
        assert!(PipelineTemplateGenerator::new(profile).is_err());
    }

    #[test]
    fn reports_infeasible_memory_with_context() {
        let profile = vec![layer(0, 1.0, 11), layer(1, 1.0, 1)];
        let mut generator = PipelineTemplateGenerator::new(profile).unwrap();
        generator.plan(1, Some(10)).unwrap();
        let error = generator.get_pipeline_template(1).unwrap_err();
        assert!(error.to_string().contains("resource count 1"));
    }
}
