import unittest
from ia_detector.gltr import GLTRAnalyzer

class TestGLTR(unittest.TestCase):
    def test_gltr_analyzer(self):
        # Integration: Real GPT-2 model load
        try:
            analyzer = GLTRAnalyzer(model_name='gpt2', device='cpu')
        except Exception:
            self.skipTest("Skipping GLTR: Model download/load failed")
            
        text = "The quick brown fox jumps over the lazy dog."
        results = analyzer.analyze(text)
        
        self.assertGreater(len(results), 0)
        self.assertIn('bucket', results[0])
        
        fractions = analyzer.get_fraction_clean(results)
        self.assertIn('Green', fractions)
        # Very common sentence should be mostly Green
        self.assertGreater(fractions.get('Green', 0), 0.4)

if __name__ == '__main__':
    unittest.main()
