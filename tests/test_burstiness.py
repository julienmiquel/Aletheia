import unittest
from ia_detector.burstiness import BurstinessAnalyzer

class TestBurstiness(unittest.TestCase):
    def test_burstiness_analyzer(self):
        # Integration: Real Spacy model load
        analyzer = BurstinessAnalyzer()
        
        text = "This is a test. It is short. The quick brown fox jumps over the lazy dog."
        res = analyzer.analyze(text)
        
        self.assertIn('burstiness_coefficient', res)
        self.assertIn('lexical_entropy', res)
        self.assertEqual(res['sentence_count'], 3)
        self.assertIsInstance(res['burstiness_coefficient'], float)
        
        # Test valid entropy
        self.assertGreater(res['lexical_entropy'], 0)

if __name__ == '__main__':
    unittest.main()
