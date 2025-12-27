import numpy as np
import warnings

# Suppress spacy warnings
warnings.filterwarnings("ignore", category=UserWarning)

class StructuralAnalyzer:
    """
    Analyzes graph-based syntactic dependency features.

    This aligns with Phase 2: Neural & Structural Features - Structural Syntax.
    It calculates metrics like average tree depth, max tree depth, and average dependency distance.
    """
    def __init__(self, model='en_core_web_sm'):
        """
        Initializes the StructuralAnalyzer. spaCy model is loaded lazily.
        """
        self.model_name = model
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
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
        Calculates structural syntax features.

        Args:
            text (str): The input text.

        Returns:
            dict: A dictionary containing structural metrics.
        """
        # Check Cache
        from ia_detector.cache import ResultCache
        if not hasattr(self, 'cache'):
            self.cache = ResultCache()

        cached = self.cache.get(text, "structural")
        if cached:
            return cached

        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            result = {
                "avg_tree_depth": 0.0,
                "max_tree_depth": 0,
                "avg_dependency_distance": 0.0
            }
            self.cache.set(text, "structural", result)
            return result

        tree_depths = []
        dep_distances = []

        for sent in sentences:
            # Tree Depth
            # Root has depth 0. We want the max depth of any token in the sentence.
            # Depth is distance from root.
            roots = [token for token in sent if token.head == token]
            if not roots:
                # Should not happen in well-formed sentences parsed by spacy
                continue

            # Since a sentence usually has one root, but let's handle if multiple (fragments)
            # Calculate depth for each token

            # Better way: iterate all tokens, count steps to root.
            # But tokens already have a head.
            # Spacy doesn't store depth directly.

            # Let's calculate max depth for this sentence.
            # We can use a recursive function or iterate.

            def get_depth(token, depth=0):
                if token.head == token:
                    return depth
                return get_depth(token.head, depth + 1)

            # However, recursion might be slow or hit limits if graph is weird (cycles? shouldn't be in dependency tree).
            # More efficient:

            # Dependency Distance: absolute difference between token index and head index.
            for token in sent:
                dist = abs(token.i - token.head.i)
                # For root, distance is 0 (token.head == token)
                dep_distances.append(dist)

            # Tree Depth calculation using graph traversal from root
            # Find the root of the sentence
            root = sent.root

            def tree_height(node):
                if not list(node.children):
                    return 0
                return 1 + max(tree_height(child) for child in node.children)

            depth = tree_height(root)
            tree_depths.append(depth)

        result = {
            "avg_tree_depth": float(np.mean(tree_depths)) if tree_depths else 0.0,
            "max_tree_depth": int(np.max(tree_depths)) if tree_depths else 0,
            "avg_dependency_distance": float(np.mean(dep_distances)) if dep_distances else 0.0
        }

        # Save to Cache
        self.cache.set(text, "structural", result)

        return result
