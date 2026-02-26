import unittest
import os
from ia_detector.llm_judge import LLMJudge
from ia_detector.semantic_consistency import SemanticConsistencyAnalyzer

class TestLatestGeminiModels(unittest.TestCase):
    def setUp(self):
        # We need an API key for integration tests
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            self.skipTest("Skipping Integration Test: GEMINI_API_KEY not found.")

        self.models_to_test = [
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b"
        ]

    def test_llm_judge_models(self):
        for model in self.models_to_test:
            with self.subTest(model=model):
                print(f"Testing LLMJudge with {model}...")
                judge = LLMJudge(model_name=model)
                res = judge.evaluate("The quick brown fox jumps over the lazy dog.")
                self.assertIn('score', res, f"Model {model} failed to return score")
                self.assertIn('reasoning', res, f"Model {model} failed to return reasoning")
                self.assertIsInstance(res['score'], (float, int))

    def test_semantic_consistency_models(self):
        for model in self.models_to_test:
            with self.subTest(model=model):
                print(f"Testing SemanticConsistencyAnalyzer with {model}...")
                analyzer = SemanticConsistencyAnalyzer(model_name=model)
                res = analyzer.analyze("The quick brown fox jumps over the lazy dog.")
                self.assertIn('consistency_score', res, f"Model {model} failed to return consistency_score")
                self.assertIn('reasoning', res, f"Model {model} failed to return reasoning")
                self.assertIsInstance(res['consistency_score'], (float, int))

if __name__ == '__main__':
    unittest.main()
