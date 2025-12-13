import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from ia_detector.cache import ResultCache

class GLTRAnalyzer:
    def __init__(self, model_name='gpt2', device=None):
        """
        Initializes the GLTR Analyzer. Model is loaded lazily.
        """
        self.cache = ResultCache()
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        
        if device:
             self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

    @property
    def model(self):
        if self._model is None:
            print(f"Loading GLTR model {self.model_name} on {self.device}...")
            self._model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def analyze(self, text):
        """
        Analyzes text to find the rank of each token in the model's prediction.
        """
        # Check Cache
        cached = self.cache.get(text, "gltr")
        if cached:
            return cached

        tokens = self.tokenizer.encode(text)
        results = []
        
        if len(tokens) < 2:
            return results

        # We need to process token by token to simulate generation context
        # This can be slow for long text, so we might want to optimize or limit length
        # For GLTR visualization, typically we look at P(token_i | tokens_0...i-1)
        
        # Batch processing optimization:
        # Construct inputs and labels
        input_ids = torch.tensor([tokens]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, :-1, :] # Predictions for next tokens (exclude last prediction)
            
            # The targets are tokens[1:]
            targets = torch.tensor(tokens[1:]).to(self.device)
            
            # For each position, find the rank of the target token
            for i, target_id in enumerate(targets):
                token_logits = logits[i]
                probs = F.softmax(token_logits, dim=-1)
                
                # Get the rank
                # We want to know where target_id falls in the sorted probabilities
                # Sorting entire vocab index is expensive. 
                # Optimization: Count how many tokens have prob > target_prob
                target_prob = probs[target_id]
                rank = (probs > target_prob).sum().item() + 1
                
                token_str = self.tokenizer.decode([target_id])
                
                # GLTR Bucketing Logic
                if rank <= 10:
                    bucket = "Green"
                elif rank <= 100:
                    bucket = "Yellow"
                elif rank <= 1000:
                    bucket = "Red"
                else:
                    bucket = "Purple"
                    
                results.append({
                    "token": token_str,
                    "rank": rank,
                    "prob": float(target_prob.item()),
                    "bucket": bucket
                })
        
        # Save to Cache 
        self.cache.set(text, "gltr", results)
                
        return results

    def get_fraction_clean(self, results):
        """
        Returns the fractions of tokens in each bucket.
        """
        if not results:
            return {}
            
        total = len(results)
        counts = {"Green": 0, "Yellow": 0, "Red": 0, "Purple": 0}
        for r in results:
            counts[r['bucket']] += 1
            
        fractions = {k: v / total for k, v in counts.items()}
        return fractions
