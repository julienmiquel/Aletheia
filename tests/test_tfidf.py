import unittest
from ia_detector.features import TfidfDetector
import os
import tempfile
import shutil

class TestTFIDF(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_tfidf_detector(self):
        model_path = os.path.join(self.test_dir, "test_tfidf.pkl")
        detector = TfidfDetector(model_path=model_path)
        
        human = ["I am a human being writing this text."]
        ai = ["I am an artificial intelligence model."]
        
        # Train
        detector.train(human, ai)
        
        # Predict
        res = detector.predict("I am a human.")
        self.assertEqual(res['prediction_label'], "Human-Written")
        
        # Save & Load
        detector.save()
        self.assertTrue(os.path.exists(model_path))
        
        detector2 = TfidfDetector(model_path=str(model_path))
        self.assertIsNotNone(detector2.pipeline)

if __name__ == '__main__':
    unittest.main()
