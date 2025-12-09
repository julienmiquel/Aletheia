import unittest
import os
from unittest.mock import MagicMock, patch
from ia_detector.ensemble import EnsembleDetector

class TestEnsemble(unittest.TestCase):
    
    @patch('ia_detector.ensemble.PerplexityCalculator')
    @patch('ia_detector.ensemble.BurstinessAnalyzer')
    @patch('ia_detector.ensemble.GLTRAnalyzer')
    @patch('ia_detector.ensemble.TfidfDetector')
    @patch('ia_detector.ensemble.LLMJudge')
    @patch('ia_detector.ensemble.SemanticConsistencyAnalyzer')
    def setUp(self, MockSemantic, MockJudge, MockTfidf, MockGLTR, MockBurst, MockPPL):
        # Setup Mocks
        self.mock_ppl = MockPPL.return_value
        self.mock_burst = MockBurst.return_value
        self.mock_gltr = MockGLTR.return_value
        self.mock_tfidf = MockTfidf.return_value
        self.mock_judge = MockJudge.return_value
        self.mock_semantic = MockSemantic.return_value
        
        # Initialize Detector (which will use the mocks)
        self.detector = EnsembleDetector(model_path='non_existent_model.pkl')

    def test_predict_heuristic_human(self):
        """Test heuristic logic for Likely Human text."""
        # Configure Mocks for "Human" signals
        self.mock_ppl.calculate.return_value = 100 # High PPL -> Human
        self.mock_burst.analyze.return_value = {'burstiness_coefficient': 0.6} # High Burst -> Human
        self.mock_gltr.analyze.return_value = {} 
        self.mock_gltr.get_fraction_clean.return_value = {'Green': 0.3} # Low Green -> Human
        self.mock_tfidf.predict.return_value = {'ai_probability': 0.1} # Low Prob -> Human
        
        result = self.detector.predict("Sample text")
        
        # Expect very low score (Human)
        # Score approx: 0 from PPL, 0 from Burst, ~25 from GLTR (0.3 is >0.2), 0.1*100*3=30 from TFIDF
        # It should be well below 50
        self.assertEqual(result['verdict'], "Human")
        self.assertLess(result['combined_score'], 50)
        self.assertIn('perplexity', result['metrics'])

    def test_predict_heuristic_ai(self):
        """Test heuristic logic for Likely AI text."""
        # Configure Mocks for "AI" signals
        self.mock_ppl.calculate.return_value = 15 # Low PPL -> AI
        self.mock_burst.analyze.return_value = 0.1 # Low Burst -> AI (handling raw float return just in case, logic handles dict)
        # Fix mock to match expected dict return for burstiness if logic expects dict
        # Actually my code does: self.burst_calc.analyze(text).get('burstiness_coefficient')
        # So I must return a dict
        self.mock_burst.analyze.return_value = {'burstiness_coefficient': 0.1}
        
        self.mock_gltr.get_fraction_clean.return_value = {'Green': 0.8} # High Green -> AI
        self.mock_tfidf.predict.return_value = {'ai_probability': 0.9} # High Prob -> AI
        
        result = self.detector.predict("Sample text")
        
        # Expect high score
        self.assertEqual(result['verdict'], "AI")
        self.assertGreater(result['combined_score'], 50)

    def test_predict_with_semantic(self):
        """Test integration of semantic analyzer."""
        # Basic human stats
        self.mock_ppl.calculate.return_value = 80
        self.mock_burst.analyze.return_value = {'burstiness_coefficient': 0.5}
        self.mock_gltr.get_fraction_clean.return_value = {'Green': 0.4}
        self.mock_tfidf.predict.return_value = {'ai_probability': 0.2}
        
        # But Semantic Analysis says AI (Inconsistent)
        self.mock_semantic.analyze.return_value = {'consistency_score': 10} # Very Inconsistent -> AI
        self.mock_judge.evaluate.return_value = {'score': 90} # Judge says AI
        
        # Run with use_semantic=True
        result = self.detector.predict("Sample text", use_semantic=True)
        
        # Semantic weights are high (4.0 each), so this should push verdict to AI likely
        # or at least raise the score significantly compared to without it
        self.assertIn('semantic_consistency', result['metrics'])
        self.assertIsNotNone(result['metrics']['semantic_consistency'])

if __name__ == '__main__':
    unittest.main()
