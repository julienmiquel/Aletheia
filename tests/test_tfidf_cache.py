import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock sklearn and other dependencies
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.feature_extraction'] = MagicMock()
sys.modules['sklearn.feature_extraction.text'] = MagicMock()
sys.modules['sklearn.linear_model'] = MagicMock()
sys.modules['sklearn.pipeline'] = MagicMock()
sys.modules['ia_detector.perplexity'] = MagicMock()
sys.modules['ia_detector.burstiness'] = MagicMock()
sys.modules['ia_detector.gltr'] = MagicMock()
sys.modules['ia_detector.binoculars'] = MagicMock()
sys.modules['ia_detector.ensemble'] = MagicMock()
sys.modules['ia_detector.llm_judge'] = MagicMock()

from ia_detector.features import TfidfDetector

class TestTfidfCaching(unittest.TestCase):
    def setUp(self):
        # Clear the class-level cache before each test
        TfidfDetector._model_cache = {}

    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open')
    def test_class_level_cache(self, mock_open, mock_pickle_load, mock_exists):
        mock_exists.return_value = True
        mock_pipeline = MagicMock()
        mock_pickle_load.return_value = mock_pipeline

        model_path = "dummy_model.pkl"

        # First instance should load from disk
        detector1 = TfidfDetector(model_path=model_path)
        p1 = detector1.pipeline

        self.assertEqual(p1, mock_pipeline)
        self.assertEqual(mock_pickle_load.call_count, 1)
        self.assertIn(str(model_path), TfidfDetector._model_cache)

        # Second instance should use cache
        detector2 = TfidfDetector(model_path=model_path)
        p2 = detector2.pipeline

        self.assertEqual(p2, mock_pipeline)
        # Should still be 1 because it was cached
        self.assertEqual(mock_pickle_load.call_count, 1)

    @patch('ia_detector.features.Pipeline')
    def test_train_updates_cache(self, mock_pipeline_class):
        mock_pipeline_instance = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline_instance

        model_path = "train_model.pkl"
        detector = TfidfDetector(model_path=model_path)

        detector.train(["human"], ["ai"])

        self.assertIn(str(model_path), TfidfDetector._model_cache)
        self.assertEqual(TfidfDetector._model_cache[str(model_path)], mock_pipeline_instance)

        # New instance should use the trained model from cache
        detector2 = TfidfDetector(model_path=model_path)
        self.assertEqual(detector2.pipeline, mock_pipeline_instance)

if __name__ == '__main__':
    unittest.main()
