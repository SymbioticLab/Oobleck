# Oobleck: Resilient Distributed Training Framework

## Evaluation Instructions

Oobleck evaluation was done in our own cluster, which can only be accessed via University network.
For artifact evaluation, we instead provide a [CameleonCloud](https://www.chameleoncloud.org/) instance with multiple NVIDIA GPUs.

Our evaluation consists of
- Functional correctness evaluation via unit tests
- Paper figure correctness evaluation via simulation results and plotter


## Prerequisite

Although we provide preconfigured environment to you, you can reconfigure the environment via the install instructions below.

### Hardware Requirements

- At least 4 NVIDIA GPUs in a node (for artifact evaluation)
- Linux (tested on Ubuntu 22.04 and Ubuntu 22.04)
- NVIDIA device driver installed
- [Anaconda](https://www.anaconda.com/download#downloads) (or its equivalent such as Miniconda or Micromamba) installed

### Install

Using `conda` on Ubuntu Linux:
```bash
conda env create -f environment.yml
conda activate oobleck
(oobleck) pip install .
...
Successfully installed oobleck-0.1.0
```

## Functional Correctness Evaluation

We provide tests for each Oobleck components based on pytest. Tests in the [`sosp23-artifact` branch](https://github.com/insujang/oobleck/tree/sosp23-artifact) are specialized in printing internal information (while normal tests don't) to help you understand its behavior and check if it matches with the paper explanation.

> To simply run all artifact tests and check if all is passed, run: `pytest tests/sosp23-artifact`.

We categorize tests as:
1. **Generating pipeline templates (Section 4.1)**: based on profiled model data (layer execution time, communication time) and number of GPUs available, we create a set of pipeline templates. Tests check if stages in each pipeline template are balanced so that in pipeline parallel execution there is no straggler stage.
2. **Instantiating pipelines (Section 4.2)**: Oobleck instantiates pipelines from the set of pipeline templates so that all available GPUs are used. For a set of pipeline templates, tests check:
    1. If Oobleck can produce pipelines for any number of GPUs
    2. If execution time of instantiated pipelines are balanced via batch distribution.
3. **Dynamic reconfiguration (Section 5)**: Oobleck dynamically reconfigures pipelines when a set of GPUs is lost by reusing the pre-generated set of pipeline templates. Tests check a set of pipeline template can be used for reconfiguration and if Oobleck can continue training with a new pipeline set.
4. **Hybrid parallelism execution**: We provide an integration test that executes two pipelines with 2 GPUs each for the same model. GPUs are only used in this test.

### Pipeline template generation

> **Known bug**: sometimes stage execution time is printed as weird number looking like uninitialized.
> This is a bug in sharing variables between Python and C++. In real execution Oobleck directly loads data in C++, so this bug only happens in the test. Please try again after a few seconds.

```python
pytest --disable-warnings --log-cli-level=INFO tests/sosp23/test_generate_pipeline_template.py 
```

We provide two fake profiled data corresponding to our evaluation models: 24 layers with equal execution time, and 32 layers with random execution time.

Sample output:
```
INFO     oobleck-sosp23:test_generate_pipeline_template.py:34 Pipeline template generation for fake_model2
        Layer execution times are random (0~5)s
        Config - ft_threshold: 1, minimum number of nodes for training (n0): 3, number of nodes: 13
        This configuration guarantees to run training with 6 ~ 13 nodes.
        Expected number of pipeline templates: 10 (pipelines with number of nodes [3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        ==============================
        
INFO     oobleck-sosp23:test_generate_pipeline_template.py:43 !!!!! TEST START !!!!!
INFO     oobleck-sosp23:test_generate_pipeline_template.py:56 Generated number of pipelines: 10 (expected: 10)
INFO     oobleck-sosp23:test_generate_pipeline_template.py:61 Checking if pipeline stages are balanced...
INFO     oobleck-sosp23:test_generate_pipeline_template.py:67     [template 0] 3 stages: [37.2 (11 layers)] [32.1 (11 layers)] [37.3 (10 layers)] 
INFO     oobleck-sosp23:test_generate_pipeline_template.py:67     [template 1] 4 stages: [24.5 (7 layers)] [27.3 (10 layers)] [27.0 (8 layers)] [27.9 (7 layers)] 
...
```

### Pipeline instantiation
```python
pytest --disable-warnings --log-cli-level=INFO tests/sosp23/test_instantiate_pipeline.py
```

After generating a set of pipeline templates, Oobleck instantiates pipelines based on the templates to assign all available GPUs. Heterogeneous pipelines have different iteration time (for the same batch size), hence Oobleck uses `batch distribution` to balance batch size and thus iteration time.

This evaluation checks if all available nodes are used, and if specified number of microbatches are distributed.
You can see how many pipeline instances per pipeline template (not all pipeline templates are used) and their estimated execution times are similar for balanced execution.

Sample output:
```
INFO     oobleck-sosp23:test_instantiate_pipeline.py:51 PIPELINE INSTANTIATION for fake_model2
        Config - ft_threshold: 3, minimum number of nodes for training (n0): 3, number of nodes: 21
        Number of pipeline templates: 5 (pipelines with number of nodes [3, 4, 5, 6, 7])
...
INFO     oobleck-sosp23:test_instantiate_pipeline.py:94 PIPELINE INSTANTIATION RESULT
[template ID 0 - 3 nodes]: 4 instances (total 12 nodes assigned). Each pipeline has 594 number of microbatches (estimated pipeline execution time: 3639.08 ms).
[template ID 1 - 4 nodes]: 1 instances (total 4 nodes assigned). Each pipeline has 758 number of microbatches (estimated pipeline execution time: 3634.72 ms).
[template ID 2 - 5 nodes]: 1 instances (total 5 nodes assigned). Each pipeline has 962 number of microbatches (estimated pipeline execution time: 3635.89 ms).

        Total number of nodes used: 21 (expected: 21)
        Total number of microbatches: 4096 (expected: 4096)
```

## Paper Figure Correctness Evaluation

We modified [Bamboo simulator](https://github.com/uclasystem/bamboo/blob/main/project_pactum/simulation/simulator.py) and used it for evaluation. We provide the link of our simulator code and Jupyter Notebooks that plot figures.