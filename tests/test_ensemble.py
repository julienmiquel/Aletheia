import unittest
import pickle
import numpy as np
import tempfile
import shutil
import os
from ia_detector.ensemble import EnsembleDetector

class DummyModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])

class TestEnsemble(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_ensemble_detector(self):
        model_path = os.path.join(self.test_dir, "ensemble_model.pkl")
        
        # Mock model object just for pickling, but test logic is integration-ish
        # We can't easily train a real Random Forest here without data, so we create a dummy class
        
        with open(model_path, 'wb') as f:
            pickle.dump(DummyModel(), f)
            
        detector = EnsembleDetector(model_path=model_path)
        features = [10, 0.1, 8.0, 0.9, 0.8] 
        
        # Test Prediction Logic
        score = detector.predict_proba(features)
        self.assertEqual(score, 80.0)
        
        # Test Dynamic Thresholding
        # 80 score -> pulled towards 50
        # 80 - 15 = 65
        score_short = detector.predict_proba(features, text_length=200)
        self.assertEqual(score_short, 65.0)

if __name__ == '__main__':
    unittest.main()
