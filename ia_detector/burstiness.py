import spacy
import numpy as np
from collections import Counter
import warnings

# Suppress spacy warnings regarding models not found if we handle it
warnings.filterwarnings("ignore", category=UserWarning)

class BurstinessAnalyzer:
    def __init__(self, model='en_core_web_sm'):
        """
        Initializes the BurstinessAnalyzer. spaCy model is loaded lazily.
        """
        self.model_name = model
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                print(f"Downloading {self.model_name}...")
                from spacy.cli import download
                download(self.model_name)
                self._nlp = spacy.load(self.model_name)
        return self._nlp

    def analyze(self, text):
        """
        Calculates burstiness based on sentence length variation and token entropy.
        
        Args:
            text (str): The input text.
            
        Returns:
            dict: A dictionary containing burstiness metrics.
        """
        # Check Cache
        from ia_detector.cache import ResultCache
        if not hasattr(self, 'cache'):
            self.cache = ResultCache()
            
        cached = self.cache.get(text, "burstiness")
        if cached:
            return cached

        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            result = {
                "sentence_count": 0,
                "mean_sentence_length": 0,
                "length_std_dev": 0,
                "burstiness_coefficient": 0,
                "lexical_entropy": 0
            }
            self.cache.set(text, "burstiness", result)
            return result
        
        # Calculate sentence lengths in tokens
        lengths = [len(sent) for sent in sentences]
        mean_len = np.mean(lengths)
        std_dev = np.std(lengths)
        
        # Coefficient of Variation (CV) as Burstiness Score
        # CV = sigma / mu
        cv = std_dev / mean_len if mean_len > 0 else 0
        
        # Entropy of word distribution (lexical diversity)
        words = [token.text.lower() for token in doc if token.is_alpha]
        word_counts = Counter(words)
        total_words = len(words)
        entropy = 0
        if total_words > 0:
            probs = [count / total_words for count in word_counts.values()]
            entropy = -sum(p * np.log(p) for p in probs)

        result = {
            "sentence_count": len(sentences),
            "mean_sentence_length": float(mean_len),
            "length_std_dev": float(std_dev),
            "burstiness_coefficient": float(cv),
            "lexical_entropy": float(entropy)
        }
        
        # Save to Cache
        self.cache.set(text, "burstiness", result)
        
        return result
