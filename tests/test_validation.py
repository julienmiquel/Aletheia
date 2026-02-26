import unittest
from unittest.mock import MagicMock
from ia_detector.validation import ValidationDataset, ValidationExample, ExperimentRunner

class MockDetector:
    def __init__(self, model_name, prompt_template=None, generation_config=None):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.config = generation_config

    def evaluate(self, text):
        return {"score": 85.5, "reasoning": "Test reasoning"}

class TestValidationFramework(unittest.TestCase):
    def test_dataset_loading(self):
        examples = [ValidationExample(text="test", label=1)]
        dataset = ValidationDataset(examples)
        self.assertEqual(len(dataset.examples), 1)

    def test_experiment_runner(self):
        dataset = ValidationDataset([
            ValidationExample(text="test1", label=1),
            ValidationExample(text="test2", label=0)
        ])

        runner = ExperimentRunner()
        result = runner.run(
            dataset=dataset,
            detector_class=MockDetector,
            model_name="mock-model"
        )

        self.assertEqual(result.model_name, "mock-model")
        self.assertEqual(len(result.details), 2)
        self.assertEqual(result.metrics['avg_score'], 85.5)
        self.assertEqual(result.metrics['sample_count'], 2)

if __name__ == '__main__':
    unittest.main()
