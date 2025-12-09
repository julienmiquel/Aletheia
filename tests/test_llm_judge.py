import unittest
import os
from ia_detector.llm_judge import LLMJudge

class TestLLMJudge(unittest.TestCase):
    def test_llm_judge_integration(self):
        if not os.getenv("GOOGLE_API_KEY"):
            self.skipTest("Skipping Integration Test: GEMINI_API_KEY not found.")
            
        # Real API Call
        judge = LLMJudge()
        
        # Use a very simple prompt to save tokens/time
        text = "The quick brown fox jumps over the lazy dog."
        
        try:
            res = judge.evaluate(text)
            self.assertIn('score', res)
            self.assertIn('reasoning', res)
            self.assertIsInstance(res['score'], float)
        except Exception as e:
            self.fail(f"LLM Judge Integration failed: {e}")

if __name__ == '__main__':
    unittest.main()
