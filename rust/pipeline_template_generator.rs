use crate::execution_result::{
    LayerExecutionResult, PipelineExecutionResult, StageExecutionResult,
};
use crate::PlannerError;
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
struct BottleneckCandidate {
    start: usize,
    end: usize,
    latency: f64,
    min_prefix_stages: usize,
    min_suffix_stages: usize,
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

    fn latency_within(&self, start: usize, end: usize, threshold: f64, inclusive: bool) -> bool {
        let ordering = self.stage(start, end).latency().total_cmp(&threshold);
        ordering == Ordering::Less || (inclusive && ordering == Ordering::Equal)
    }

    fn stage_allowed(
        &self,
        start: usize,
        end: usize,
        threshold: f64,
        inclusive: bool,
        device_memory_bytes: Option<u64>,
    ) -> bool {
        self.stage(start, end).fits_one_microbatch(device_memory_bytes)
            && self.latency_within(start, end, threshold, inclusive)
    }

    /// Minimum number of contiguous stages needed for [start, end) under a
    /// bottleneck bound. Non-negative additive time and memory make the greedy
    /// longest-feasible-stage choice optimal. Any count up to the number of
    /// layers is then reachable by splitting stages.
    fn min_segments(
        &self,
        start: usize,
        end: usize,
        threshold: f64,
        inclusive: bool,
        device_memory_bytes: Option<u64>,
    ) -> Option<usize> {
        let mut position = start;
        let mut count = 0;
        while position < end {
            let mut farthest = None;
            for next in (position + 1)..=end {
                if self.stage_allowed(
                    position,
                    next,
                    threshold,
                    inclusive,
                    device_memory_bytes,
                ) {
                    farthest = Some(next);
                } else {
                    break;
                }
            }
            position = farthest?;
            count += 1;
        }
        Some(count)
    }

    /// Return the lexicographically smallest exact partition under a latency
    /// bound. This is used only to materialize a winning paper candidate; the
    /// feasibility bounds are shared by every requested template size.
    fn lexicographic_partition(
        &self,
        start: usize,
        end: usize,
        count: usize,
        threshold: f64,
        inclusive: bool,
        device_memory_bytes: Option<u64>,
    ) -> Option<Vec<usize>> {
        if count == 0 {
            return (start == end).then(Vec::new);
        }
        if count > end - start {
            return None;
        }

        let mut starts = Vec::with_capacity(count);
        let mut position = start;
        for stage_index in 0..count {
            starts.push(position);
            let remaining = count - stage_index - 1;
            if remaining == 0 {
                return self
                    .stage_allowed(
                        position,
                        end,
                        threshold,
                        inclusive,
                        device_memory_bytes,
                    )
                    .then_some(starts);
            }

            let latest = end - remaining;
            let mut chosen = None;
            for next in (position + 1)..=latest {
                if !self.stage_allowed(
                    position,
                    next,
                    threshold,
                    inclusive,
                    device_memory_bytes,
                ) {
                    break;
                }
                let minimum = self.min_segments(
                    next,
                    end,
                    threshold,
                    inclusive,
                    device_memory_bytes,
                );
                if minimum.is_some_and(|minimum| minimum <= remaining)
                    && remaining <= end - next
                {
                    chosen = Some(next);
                    break;
                }
            }
            position = chosen?;
        }
        None
    }

    fn bottleneck_candidates(
        &self,
        device_memory_bytes: Option<u64>,
    ) -> Vec<BottleneckCandidate> {
        let num_layers = self.layer_execution_results.len();
        let mut candidates = Vec::new();
        for start in 0..num_layers {
            for end in (start + 1)..=num_layers {
                let stage = self.stage(start, end);
                if !stage.fits_one_microbatch(device_memory_bytes) {
                    continue;
                }
                let latency = stage.latency();
                let Some(min_prefix_stages) = self.min_segments(
                    0,
                    start,
                    latency,
                    true,
                    device_memory_bytes,
                ) else {
                    continue;
                };
                // k* is the rightmost bottleneck, so following stages must be
                // strictly faster while preceding stages may tie it.
                let Some(min_suffix_stages) = self.min_segments(
                    end,
                    num_layers,
                    latency,
                    false,
                    device_memory_bytes,
                ) else {
                    continue;
                };
                candidates.push(BottleneckCandidate {
                    start,
                    end,
                    latency,
                    min_prefix_stages,
                    min_suffix_stages,
                });
            }
        }
        candidates
    }

    fn materialize_candidate(
        &self,
        candidate: BottleneckCandidate,
        prefix_stages: usize,
        total_stages: usize,
        device_memory_bytes: Option<u64>,
    ) -> Option<PipelineExecutionResult> {
        let suffix_stages = total_stages - prefix_stages - 1;
        let mut starts = self.lexicographic_partition(
            0,
            candidate.start,
            prefix_stages,
            candidate.latency,
            true,
            device_memory_bytes,
        )?;
        starts.push(candidate.start);
        starts.extend(self.lexicographic_partition(
            candidate.end,
            self.layer_execution_results.len(),
            suffix_stages,
            candidate.latency,
            false,
            device_memory_bytes,
        )?);

        let ends = starts
            .iter()
            .copied()
            .skip(1)
            .chain(std::iter::once(self.layer_execution_results.len()));
        let mut stages = starts
            .iter()
            .copied()
            .zip(ends)
            .map(|(start, end)| self.stage(start, end).clone());
        let first = stages
            .next()
            .expect("a materialized candidate always contains its bottleneck stage");
        let mut result = PipelineExecutionResult::from_stage(first);
        for stage in stages {
            let right = PipelineExecutionResult::from_stage(stage);
            result = PipelineExecutionResult::combine(&result, &right);
        }
        debug_assert_eq!(result.kstar, prefix_stages);
        Some(result)
    }

    /// Generate fixed-TP templates using Oobleck Section 4.1.2, Equations 1--4.
    ///
    /// The paper jointly divides layers and devices. The refactored runtime
    /// treats one complete fixed-TP node as an indivisible logical device, so
    /// S=d=n and the within-node GPU split m disappears. For every possible
    /// rightmost bottleneck stage this method derives T1, T2, and T3 exactly,
    /// checks whether its left and right subproblems can be conquered, and
    /// chooses the minimum paper iteration time with Nb=4S.
    ///
    /// Stage metrics and bottleneck feasibility summaries are built once and
    /// reused for every resource count in this invocation, preserving the
    /// cross-template cache reuse of the artifact implementation.
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

        let candidates = self.bottleneck_candidates(device_memory_bytes);
        let total_work = self.stage(0, num_layers).latency();
        self.pipeline_execution_results = vec![None; max_stages + 1];

        for total_stages in 1..=max_stages {
            let mut best: Option<PipelineExecutionResult> = None;
            for candidate in candidates.iter().copied() {
                let prefix_lower = candidate
                    .min_prefix_stages
                    .max(total_stages.saturating_sub(1 + (num_layers - candidate.end)));
                let prefix_upper = candidate.start.min(
                    total_stages
                        .saturating_sub(1 + candidate.min_suffix_stages),
                );
                if prefix_lower > prefix_upper || prefix_lower >= total_stages {
                    continue;
                }

                // Increasing k* adds one bottleneck latency to Equation 2.
                // Only a zero-latency bottleneck can tie across several k*s.
                let prefix_range = if candidate.latency == 0.0 {
                    prefix_lower..=prefix_upper
                } else {
                    prefix_lower..=prefix_lower
                };
                for prefix_stages in prefix_range {
                    let t3 = self.stage(candidate.start, num_layers).latency();
                    let predicted = total_work
                        + (3 * total_stages + prefix_stages - 1) as f64
                            * candidate.latency
                        + t3;
                    if best.as_ref().is_some_and(|current| {
                        predicted.total_cmp(&current.planning_iteration_time())
                            == Ordering::Greater
                    }) {
                        continue;
                    }
                    let Some(result) = self.materialize_candidate(
                        candidate,
                        prefix_stages,
                        total_stages,
                        device_memory_bytes,
                    ) else {
                        continue;
                    };
                    let replace = best.as_ref().map_or(true, |current| {
                        result
                            .planning_iteration_time()
                            .total_cmp(&current.planning_iteration_time())
                            .then_with(|| {
                                result
                                    .stages
                                    .iter()
                                    .map(|stage| stage.layers.0)
                                    .cmp(current.stages.iter().map(|stage| stage.layers.0))
                            })
                            == Ordering::Less
                    });
                    if replace {
                        best = Some(result);
                    }
                }
            }
            self.pipeline_execution_results[total_stages] = best;
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

    fn paper_time(latencies: &[f64], starts: &[usize]) -> f64 {
        let ends = starts
            .iter()
            .copied()
            .skip(1)
            .chain(std::iter::once(latencies.len()));
        let work: Vec<f64> = starts
            .iter()
            .copied()
            .zip(ends)
            .map(|(start, end)| latencies[start..end].iter().sum())
            .collect();
        let kstar = work
            .iter()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .total_cmp(right.1)
                    .then_with(|| left.0.cmp(&right.0))
            })
            .unwrap()
            .0;
        let t1: f64 = work.iter().sum();
        let t3: f64 = work[kstar..].iter().sum();
        let t2 = (3 * starts.len() + kstar - 1) as f64 * work[kstar];
        t1 + t2 + t3
    }

    fn brute_force(latencies: &[f64], stages: usize) -> (f64, Vec<usize>) {
        fn visit(
            latencies: &[f64],
            stages: usize,
            next: usize,
            starts: &mut Vec<usize>,
            best: &mut Option<(f64, Vec<usize>)>,
        ) {
            if starts.len() == stages {
                let candidate = (paper_time(latencies, starts), starts.clone());
                if best.as_ref().map_or(true, |current| {
                    candidate
                        .0
                        .total_cmp(&current.0)
                        .then_with(|| candidate.1.cmp(&current.1))
                        == Ordering::Less
                }) {
                    *best = Some(candidate);
                }
                return;
            }
            let remaining_starts = stages - starts.len();
            let last = latencies.len() - remaining_starts;
            for start in next..=last {
                starts.push(start);
                visit(latencies, stages, start + 1, starts, best);
                starts.pop();
            }
        }

        let mut best = None;
        let mut starts = vec![0];
        visit(latencies, stages, 1, &mut starts, &mut best);
        best.unwrap()
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
    fn matches_an_exhaustive_paper_objective_oracle() {
        let latencies = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let generator = prepare(&latencies, 4, None).unwrap();
        for stage_count in 1..=4 {
            let result = generator.get_pipeline_template(stage_count).unwrap();
            let expected = brute_force(&latencies, stage_count as usize);
            assert_eq!(
                result.stages.iter().map(|stage| stage.layers.0 as usize).collect::<Vec<_>>(),
                expected.1
            );
            assert_eq!(result.planning_iteration_time(), expected.0);
        }
    }

    #[test]
    fn fixes_the_legacy_single_result_cache_counterexample() {
        let latencies = [91.0, 45.0, 29.0, 75.0, 73.0, 66.0, 99.0, 81.0, 83.0, 83.0];
        let generator = prepare(&latencies, 6, None).unwrap();
        let result = generator.get_pipeline_template(6).unwrap();
        let expected = brute_force(&latencies, 6);
        assert_eq!(
            result.stages.iter().map(|stage| stage.layers.0 as usize).collect::<Vec<_>>(),
            expected.1
        );
        assert_eq!(result.planning_iteration_time(), expected.0);
    }

    #[test]
    fn follows_equations_one_through_four() {
        let generator = prepare(&[2.0, 3.0, 4.0], 2, None).unwrap();
        let result = generator.get_pipeline_template(2).unwrap();
        assert_eq!(result.t1, 9.0);
        assert_eq!(
            result.t3,
            result.stages[result.kstar..]
                .iter()
                .map(StageExecutionResult::latency)
                .sum::<f64>()
        );
        assert_eq!(result.planning_iteration_time(), result.iteration_time(8));
    }

    #[test]
    fn uses_lexicographically_smallest_cuts_for_equal_paper_costs() {
        let generator = prepare(&[0.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(
            ranges(&generator.get_pipeline_template(2).unwrap()),
            vec![(0, 1), (1, 3)]
        );
    }

    #[test]
    fn excludes_paper_candidates_that_do_not_fit_memory() {
        let profile = vec![layer(0, 1.0, 8), layer(1, 5.0, 5), layer(2, 6.0, 1)];
        let mut generator = PipelineTemplateGenerator::new(profile).unwrap();
        generator.plan(2, Some(10)).unwrap();
        let result = generator.get_pipeline_template(2).unwrap();
        assert_eq!(ranges(&result), vec![(0, 1), (1, 3)]);
        assert_eq!(result.max_microbatches(Some(10)), Some(1));
    }

    #[test]
    fn one_shared_cache_produces_every_requested_template_size() {
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
