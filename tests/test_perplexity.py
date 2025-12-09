import unittest
from ia_detector.perplexity import PerplexityCalculator

class TestPerplexity(unittest.TestCase):
    def test_perplexity_calculator(self):
        # Integration: Real GPT-2 model load
        try:
            calculator = PerplexityCalculator(model_id='gpt2', device='cpu')
        except Exception:
            self.skipTest("Skipping Perplexity: Model download/load failed")
            
        text = "The quick brown fox jumps over the lazy dog."
        ppl = calculator.calculate(text)
        
        self.assertIsInstance(ppl, float)
        self.assertGreater(ppl, 0)
        self.assertLess(ppl, 1000)

if __name__ == '__main__':
    unittest.main()
