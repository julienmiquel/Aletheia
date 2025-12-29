import unittest
import torch
from ia_detector.structural import StructuralAnalyzer
from ia_detector.surrogate_ppl import SurrogatePPLDetector
from ia_detector.vector_style import VectorStyleAnalyzer
from ia_detector.ensemble import EnsembleDetector

class TestNewFeatures(unittest.TestCase):
    def test_structural_analyzer(self):
        analyzer = StructuralAnalyzer()
        text = "This is a simple sentence."
        result = analyzer.analyze(text)
        self.assertIn("avg_tree_depth", result)
        self.assertIn("max_tree_depth", result)
        self.assertIn("avg_dependency_distance", result)
        self.assertGreater(result["avg_tree_depth"], 0)

    def test_surrogate_ppl(self):
        # Use CPU and minimal config for speed in test
        # Note: loading models takes time, so this test might be slow
        if not torch.cuda.is_available():
            # Skip if no GPU, might be too slow for CI?
            # But let's run it once to verify logic.
            pass

        analyzer = SurrogatePPLDetector(perturbation_model_id='t5-small', scoring_model_id='gpt2', device='cpu')
        text = "This is a test sentence."
        # Reduce perturbations for speed
        result = analyzer.analyze(text, n_perturbations=1)
        self.assertIn("surrogate_ppl_score", result)
        self.assertIn("is_ai_likely", result)

    def test_vector_style(self):
        analyzer = VectorStyleAnalyzer(model_name='all-MiniLM-L6-v2')
        text = "This is a sentence. This is another one."
        result = analyzer.analyze(text)
        self.assertIn("avg_coherence", result)
        self.assertIn("coherence_variance", result)

    def test_ensemble_integration(self):
        # Mock detectors to avoid loading heavy models?
        # Ideally yes, but here we integration test.
        # We can mock the sub-detectors in the ensemble instance.

        ensemble = EnsembleDetector()

        # Mock methods
        ensemble.ppl_calc.calculate = lambda x: 25.0
        ensemble.burst_calc.analyze = lambda x: {"burstiness_coefficient": 0.3}
        ensemble.gltr_calc.analyze = lambda x: []
        ensemble.gltr_calc.get_fraction_clean = lambda x: {"Green": 0.8}
        ensemble.tfidf_detector.predict = lambda x: {"ai_probability": 0.9}

        ensemble.structural_calc.analyze = lambda x: {"avg_tree_depth": 4.0}
        ensemble.surrogate_ppl_calc.analyze = lambda x: {"surrogate_ppl_score": 2.0}
        ensemble.vector_style_calc.analyze = lambda x: {"avg_coherence": 0.8}

        result = ensemble.predict("Test text", use_semantic=False)

        metrics = result['metrics']
        self.assertIn("avg_tree_depth", metrics)
        self.assertIn("surrogate_ppl_score", metrics)
        self.assertIn("avg_coherence", metrics)

        # Check if score increased due to AI-like mock values
        # score should be > 60
        self.assertEqual(result['verdict'], "AI")

if __name__ == '__main__':
    unittest.main()
