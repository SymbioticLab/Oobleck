import pytest
import random
import json
from pathlib import Path
import shutil


@pytest.fixture(scope="module")
def dummy_model():
    model_name = "testmodel"
    model_tag = "dev"
    num_layers = 32

    layers = []
    allreduce_across_nodes = []
    allreduce_in_node = []
    for _ in range(num_layers):
        layers.append(
            {
                "forward": random.random(),
                "backward": random.random() * 3,
                "mem_required": [1024.0, 1024.0],
            }
        )
        allreduce_across_nodes.append(
            {"1": random.random() * 4, "4": random.random() * 4}
        )
        allreduce_in_node.append({"1": random.random(), "4": random.random()})

    directory = Path(f"/tmp/oobleck/profiles/{model_name}-{model_tag}")
    directory.mkdir(parents=True, exist_ok=True)

    with directory.joinpath("mb1.json").open("w") as f:
        json.dump(layers, f)
    with directory.joinpath("mb4.json").open("w") as f:
        json.dump(layers, f)
    with directory.joinpath("allreduce_in_node.json").open("w") as f:
        json.dump(allreduce_in_node, f)
    with directory.joinpath("allreduce_across_nodes.json").open("w") as f:
        json.dump(allreduce_across_nodes, f)

    yield model_name, model_tag, num_layers

    shutil.rmtree(directory)


def test_create_pipeline_templates_onenode(dummy_model, pipeline_template_generator):
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        dummy_model[0],
        dummy_model[1],
        1,  # microbatch_size
        (1, 1),  # num nodes range
        1,
    )
    assert len(pipeline_templates) == 1
    assert pipeline_templates[0].get_num_nodes() == 1
    assert pipeline_templates[0].get_num_gpus_per_node() == 1
    assert len(pipeline_templates[0].get_stages()) == 1
    assert pipeline_templates[0].get_iteration_time() > 0


def test_create_pipeline_templates_maxnode(dummy_model, pipeline_template_generator):
    num_nodes = dummy_model[2]
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        dummy_model[0],
        dummy_model[1],
        1,  # microbatch_size
        (num_nodes, num_nodes),  # num nodes range
        1,
    )
    assert len(pipeline_templates) == 1
    assert pipeline_templates[0].get_num_nodes() == num_nodes
    assert pipeline_templates[0].get_num_gpus_per_node() == 1
    assert len(pipeline_templates[0].get_stages()) == num_nodes
    assert pipeline_templates[0].get_iteration_time() > 0


def test_create_pipeline_templates_toomanynodes(
    dummy_model, pipeline_template_generator
):
    num_nodes = dummy_model[2] + 1
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        dummy_model[0],
        dummy_model[1],
        1,  # microbatch_size
        (num_nodes, num_nodes),  # num nodes range
        1,
    )
    assert len(pipeline_templates) == 0


def test_create_pipeline_templates_noderange(dummy_model, pipeline_template_generator):
    num_nodes = dummy_model[2]
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        dummy_model[0],
        dummy_model[1],
        1,  # microbatch_size
        (1, dummy_model[2]),  # num nodes range
        1,
    )
    assert 0 < len(pipeline_templates) <= num_nodes
    assert 0 < pipeline_templates[0].get_num_nodes() <= num_nodes
    assert len(pipeline_templates[0].get_stages()) == 1
    assert len(pipeline_templates[-1].get_stages()) == num_nodes
    for pipeline_template in pipeline_templates:
        assert pipeline_templates[0].get_num_gpus_per_node() == 1
        assert len(pipeline_template.get_stages()) > 0
        assert pipeline_template.get_iteration_time() > 0


def test_create_pipeline_templates_multimicrobatch(
    dummy_model, pipeline_template_generator
):
    num_nodes = dummy_model[2]
    pipeline_templates = pipeline_template_generator.create_pipeline_templates(
        dummy_model[0],
        dummy_model[1],
        4,  # microbatch_size
        (1, num_nodes),  # num nodes range
        1,
    )
    assert 0 < len(pipeline_templates) <= num_nodes
    assert 0 < pipeline_templates[0].get_num_nodes() <= num_nodes
    assert len(pipeline_templates[0].get_stages()) == 1
    assert len(pipeline_templates[-1].get_stages()) == num_nodes
    for pipeline_template in pipeline_templates:
        assert pipeline_templates[0].get_num_gpus_per_node() == 1
        assert len(pipeline_template.get_stages()) > 0
        assert pipeline_template.get_iteration_time() > 0


def test_create_pipeline_templates_no_microbatch_profile(
    dummy_model, pipeline_template_generator
):
    with pytest.raises(Exception):
        pipeline_template_generator.create_pipeline_templates(
            dummy_model[0],
            dummy_model[1],
            2,  # microbatch_size
            (1, 1),  # num nodes range
            1,
        )


@pytest.mark.skip(reason="TODO")
def test_create_pipeline_templates_fsdp(gpt2_model, pipeline_template_generator):
    assert False
    generator = PipelineTemplateGenerator()
    pipeline_templates = generator.create_pipeline_templates(
        gpt2_model.model_name,
        gpt2_model.model_tag,
        1,  # microbatch_size
        (2, 10),  # num nodes range
        4,
    )
