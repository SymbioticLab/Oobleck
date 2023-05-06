import unittest
from pipeline_template import PipelineTemplate, PipelineTemplateGenerator
import faulthandler


class TestOobleckPipelineTemplate(unittest.TestCase):
    def test_create_generator(self):
        generator: PipelineTemplateGenerator = PipelineTemplateGenerator()
        self.assertIsNotNone(generator)

    def test_create_templates(self):
        generator: PipelineTemplateGenerator = PipelineTemplateGenerator()
        template: PipelineTemplate = generator.create_pipeline_templates(
            "gpt2", "2.7b", 2, (3, 11), 1
        )
        self.assertTrue(isinstance(template, list))


if __name__ == "__main__":
    faulthandler.enable()
    unittest.main()
