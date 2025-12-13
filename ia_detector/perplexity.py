import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
from ia_detector.cache import ResultCache

class PerplexityCalculator:
    def __init__(self, model_id='gpt2', device=None):
        """
        Initializes the PerplexityCalculator. Models are loaded lazily.
        Args:
            model_id (str): The Hugging Face model ID (default: 'gpt2').
            device (str): Computation device ('cuda', 'mps', or 'cpu').
        """
        self.cache = ResultCache()
        self.model_id = model_id
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
            print(f"Loading model {self.model_id} on {self.device}...")
            self._model = GPT2LMHeadModel.from_pretrained(self.model_id).to(self.device)
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = GPT2TokenizerFast.from_pretrained(self.model_id)
        return self._tokenizer

    @property
    def max_length(self):
        return self.model.config.n_positions

    def calculate(self, text, stride=512):
        """
        Calculates the perplexity of a text using a sliding window approach.
        """
        # Check Cache
        cached_result = self.cache.get(text, "perplexity")
        if cached_result is not None:
            return cached_result

        encodings = self.tokenizer(text, return_tensors='pt')
        seq_len = encodings.input_ids.size(1)
        
        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0
        
        # Sliding window loop
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Computing Perplexity", disable=True):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            
            # Mask the context tokens so they don't contribute to the loss
            # We only want to evaluate the likelihood of the *new* tokens in this window
            target_ids[:, :-trg_len] = -100 
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                
                # outputs.loss is the average NLL per token. 
                # We multiply by the number of valid tokens to get the sum.
                neg_log_likelihood = outputs.loss
                
                # Calculate the number of tokens contributing to the loss calculation
                num_valid_tokens = (target_ids != -100).sum().item()
                
                nll_sum += neg_log_likelihood * num_valid_tokens
                n_tokens += num_valid_tokens
                
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        if n_tokens == 0:
            return float('nan')

        # Perplexity = exp(average_NLL)
        avg_nll = nll_sum / n_tokens
        ppl = torch.exp(avg_nll)
        result = ppl.item()
        
        # Save to Cache
        self.cache.set(text, "perplexity", result)
        
        return result
