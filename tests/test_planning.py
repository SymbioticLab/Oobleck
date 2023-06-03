from pipeline_template import PipelineTemplate, PipelineTemplateGenerator
import faulthandler
import time


# class TestOobleckPipelineTemplate(unittest.TestCase):
#     @unittest.skip("")
#     def test_create_generator(self):
#         generator: PipelineTemplateGenerator = PipelineTemplateGenerator()
#         self.assertIsNotNone(generator)

#     def test_create_templates(self):
#         start = time.time()
#         generator: PipelineTemplateGenerator = PipelineTemplateGenerator()
#         template: PipelineTemplate = generator.create_pipeline_templates(
#             "gpt2", "2.7b", 2, (3, 12), 1
#         )
#         end = time.time()
#         print("Time to create templates: ", end - start)
#         self.assertTrue(isinstance(template, list))


# if __name__ == "__main__":
#     faulthandler.enable()
#     unittest.main()
