import pickle
import numpy as np
import joblib
import os
from ia_detector.perplexity import PerplexityCalculator
from ia_detector.burstiness import BurstinessAnalyzer
from ia_detector.gltr import GLTRAnalyzer
from ia_detector.features import TfidfDetector
from ia_detector.llm_judge import LLMJudge
from ia_detector.semantic_consistency import SemanticConsistencyAnalyzer
from ia_detector import config

class EnsembleDetector:
    def __init__(self, model_path=config.ENSEMBLE_MODEL_PATH):
        self.model_path = model_path
        """
        Initializes the EnsembleDetector and all sub-detectors.
        """
        self.model = None
        self._load_model(model_path)
        
        # Initialize Sub-Detectors
        print("Ensemble: Initializing Perplexity...")
        self.ppl_calc = PerplexityCalculator()
        
        print("Ensemble: Initializing Burstiness...")
        self.burst_calc = BurstinessAnalyzer()
        
        print("Ensemble: Initializing GLTR...")
        self.gltr_calc = GLTRAnalyzer()
        
        print("Ensemble: Initializing TF-IDF...")
        self.tfidf_detector = TfidfDetector()
        
        # Semantic Models (Optional/Costly)
        print("Ensemble: Initializing Semantic Models (Lazy load)...")
        self.llm_judge = LLMJudge()
        self.semantic_analyzer = SemanticConsistencyAnalyzer()

    def _load_model(self, model_path):
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load ensemble model: {e}")

    def predict(self, text, use_semantic=False):
        """
        Runs all detectors and returns a comprehensive verdict.
        
        Args:
            text (str): Input text.
            use_semantic (bool): Whether to use expensive LLM-based checks.
            
        Returns:
            dict: Full analysis report.
        """
        metrics = {}
        
        # 1. Perplexity
        try:
            metrics['perplexity'] = self.ppl_calc.calculate(text)
        except Exception as e:
            print(f"Ensemble PPL Error: {e}")
            metrics['perplexity'] = None

        # 2. Burstiness
        try:
            burst_res = self.burst_calc.analyze(text)
            metrics['burstiness'] = burst_res.get('burstiness_coefficient')
            metrics['lexical_entropy'] = burst_res.get('lexical_entropy')
        except: 
            metrics['burstiness'] = None
            metrics['lexical_entropy'] = 0.0

        # 3. GLTR
        try:
            gltr_res = self.gltr_calc.analyze(text)
            metrics['gltr_green'] = self.gltr_calc.get_fraction_clean(gltr_res).get('Green')
        except: metrics['gltr_green'] = None

        # 4. TF-IDF
        try:
            tfidf_res = self.tfidf_detector.predict(text)
            metrics['tfidf_prob'] = tfidf_res.get('ai_probability')
        except: metrics['tfidf_prob'] = None

        # 5. Semantic (Optional)
        metrics['semantic_consistency'] = None
        metrics['llm_judge_score'] = None
        if use_semantic:
            try:
                sem_res = self.semantic_analyzer.analyze(text)
                metrics['semantic_consistency'] = sem_res.get('consistency_score')
                metrics['coherence_metric'] = sem_res.get('coherence_metric')
                metrics['llm_judge_score'] = self.llm_judge.evaluate(text).get('score')
            except Exception as e:
                print(f"Ensemble Semantic Error: {e}")

        # Final Scoring
        if self.model:
            # ML-based prediction
            score = self._predict_ml(metrics, len(text))
        else:
            # Heuristic-based prediction
            score = self._predict_heuristic(metrics)

        return {
            "combined_score": score,
            "verdict": "AI" if score > 50 else "Human",
            "confidence": "High" if abs(score - 50) > 30 else "Low",
            "metrics": metrics
        }

    def _predict_ml(self, metrics, length):
        # Fallback to heuristic if ML inputs are missing
        if None in [metrics['perplexity'], metrics['burstiness'], metrics['gltr_green'], metrics['tfidf_prob']]:
            return self._predict_heuristic(metrics)
            
        features = [
            metrics['perplexity'],
            metrics['burstiness'],
            metrics.get('lexical_entropy', 0), # Added entropy
            metrics['gltr_green'],
            metrics['tfidf_prob']
        ]
        
        try:
            return self.model.predict_proba([features])[0][1] * 100
        except:
            return self._predict_heuristic(metrics)

    def _predict_heuristic(self, metrics):
        score = 0
        weight = 0
        
        # PPL: Low is AI (<20), High is Human (>80)
        p = metrics.get('perplexity')
        if p is not None:
            # 20 -> 100, 80 -> 0
            s = max(0, min(100, (80 - p) * (100/60)))
            score += s * 2.5
            weight += 2.5

        # Burstiness: Low is AI (<0.2), High is Human (>0.5)
        b = metrics.get('burstiness')
        if b is not None:
            # 0.2 -> 100, 0.5 -> 0
            s = max(0, min(100, (0.5 - b) * (100/0.3)))
            score += s * 2.0
            weight += 2.0

        # GLTR: High Green is AI (>0.6)
        g = metrics.get('gltr_green')
        if g is not None:
            # 0.2 -> 0, 0.6 -> 100
            s = max(0, min(100, (g - 0.2) * (100/0.4)))
            score += s * 1.5
            weight += 1.5

        # TF-IDF: Probability is 0-1
        t = metrics.get('tfidf_prob')
        if t is not None:
            score += t * 100 * 3.0 # Strongest stylistic signal
            weight += 3.0
            
        # Semantic: High Consistent is Human (100 -> 0% AI), Low is AI (0 -> 100% AI)
        s = metrics.get('semantic_consistency')
        if s is not None:
            # Score is "Consistency" (Human-ness). Invert for AI likelihood.
            ai_likelihood = 100 - s
            score += ai_likelihood * 4.0 # Very High Weight if run
            weight += 4.0

        # LLM Judge: Returns AI Score (0-100)
        j = metrics.get('llm_judge_score')
        if j is not None:
            score += j * 4.0 # Very High Weight
            weight += 4.0

        if weight == 0:
            return 50.0
            
        return score / weight
