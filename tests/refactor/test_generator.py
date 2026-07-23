from oobleck.planning import LayerExecutionResult, create_pipeline_templates


def test_template_generator_covers_uneven_global_ranges():
    layers = [
        LayerExecutionResult(index, f"layer.{index}", index + 1, index + 1, 10)
        for index in range(6)
    ]
    templates = create_pipeline_templates("toy", layers, [1, 2, 3], 1)
    assert set(templates) == {1, 2, 3}
    for stages, template in templates.items():
        assert template.num_stages == stages
        covered = [index for start, end in template.layer_ranges for index in range(start, end)]
        assert covered == list(range(6))
    assert templates[2].layer_ranges[0] != (0, 3)
