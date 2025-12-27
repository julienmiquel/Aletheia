import numpy as np
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class VectorStyleAnalyzer:
    """
    Analyzes sentence embedding trajectories (Narrative Flow).
    This aligns with Phase 2: Neural & Structural Features - Vector-Based Style.

    It calculates the coherence (cosine similarity) between consecutive sentences.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the VectorStyleAnalyzer. Model is loaded lazily.
        """
        self.model_name = model_name
        self._model = None
        self._nlp = None

    @property
    def model(self):
        if self._model is None:
            print(f"Loading SentenceTransformer {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def analyze(self, text):
        """
        Calculates vector-based style metrics.

        Args:
            text (str): The input text.

        Returns:
            dict: Metrics including average coherence and coherence variance.
        """
        # Check Cache
        from ia_detector.cache import ResultCache
        if not hasattr(self, 'cache'):
            self.cache = ResultCache()

        cached = self.cache.get(text, "vector_style")
        if cached:
            return cached

        # Segment sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

        if len(sentences) < 2:
            result = {
                "avg_coherence": 0.0,
                "coherence_variance": 0.0,
                "min_coherence": 0.0
            }
            self.cache.set(text, "vector_style", result)
            return result

        # Compute embeddings
        embeddings = self.model.encode(sentences)

        # Calculate cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            vec1 = embeddings[i]
            vec2 = embeddings[i+1]

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 > 0 and norm2 > 0:
                sim = np.dot(vec1, vec2) / (norm1 * norm2)
                similarities.append(sim)
            else:
                similarities.append(0.0)

        if not similarities:
             result = {
                "avg_coherence": 0.0,
                "coherence_variance": 0.0,
                "min_coherence": 0.0
            }
        else:
            result = {
                "avg_coherence": float(np.mean(similarities)),
                "coherence_variance": float(np.var(similarities)),
                "min_coherence": float(np.min(similarities))
            }

        self.cache.set(text, "vector_style", result)

        return result
