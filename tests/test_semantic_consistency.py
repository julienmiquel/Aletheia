import unittest
import os
from unittest.mock import MagicMock, patch
from ia_detector.semantic_consistency import SemanticConsistencyAnalyzer

class TestSemanticConsistency(unittest.TestCase):
    def test_initialization(self):
        """Test API key detection."""
        # Mock environment
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            analyzer = SemanticConsistencyAnalyzer()
            self.assertIsNotNone(analyzer.client)

    @patch('ia_detector.semantic_consistency.genai.Client')
    def test_analyze_consistent(self, mock_client_cls):
        """Test analysis of consistent text."""
        # Setup Mock
        mock_instance = mock_client_cls.return_value
        mock_response = MagicMock()
        mock_response.text = '{"consistency_score": 95, "reasoning": "Text is logical."}'
        mock_instance.models.generate_content.return_value = mock_response
        
        analyzer = SemanticConsistencyAnalyzer()
        # Manually set client to mock instance because __init__ might fail if no real key
        analyzer.client = mock_instance 
        
        result = analyzer.analyze("The sky is blue. It is a sunny day.")
        
        self.assertEqual(result['consistency_score'], 95)
        self.assertEqual(result['reasoning'], "Text is logical.")

    @patch('ia_detector.semantic_consistency.genai.Client')
    def test_analyze_inconsistent(self, mock_client_cls):
        """Test analysis of contradictory text."""
        # Setup Mock
        mock_instance = mock_client_cls.return_value
        mock_response = MagicMock()
        mock_response.text = '{"consistency_score": 10, "reasoning": "Contradiction found: Red car vs Blue car."}'
        mock_instance.models.generate_content.return_value = mock_response
        
        analyzer = SemanticConsistencyAnalyzer()
        analyzer.client = mock_instance
        
        result = analyzer.analyze("The car was red. Later, the blue car drove away.")
        
        self.assertEqual(result['consistency_score'], 10)
        self.assertIn("Contradiction", result['reasoning'])

    def test_integration_real_call(self):
        """Integration test with real API if key exists."""
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            self.skipTest("Skipping Integration Test: No API KEY found.")

        analyzer = SemanticConsistencyAnalyzer()
        # Use a text that is obviously consistent
        text = "The quick brown fox jumps over the lazy dog."
        result = analyzer.analyze(text)
        
        # Real models might vary, but valid JSON should be returned
        self.assertIn('consistency_score', result)
        self.assertIn('reasoning', result)
        self.assertIsInstance(result['consistency_score'], float)

if __name__ == '__main__':
    unittest.main()
