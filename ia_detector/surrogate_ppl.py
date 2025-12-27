import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
import numpy as np
import re
from tqdm import tqdm

class SurrogatePPLDetector:
    """
    Implements a simplified DetectGPT-like method using perturbation discrepancy.
    This aligns with Phase 2: Neural & Structural Features - Surrogate PPL.

    It perturbs the text using a mask-filling model (T5) and compares the perplexity
    of the original text vs the perturbed text using a scoring model (GPT-2).
    """
    def __init__(self, perturbation_model_id='t5-small', scoring_model_id='gpt2', device=None):
        from ia_detector.cache import ResultCache
        self.cache = ResultCache()

        self.perturbation_model_id = perturbation_model_id
        self.scoring_model_id = scoring_model_id

        if device:
             self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self._perturbation_model = None
        self._perturbation_tokenizer = None
        self._scoring_model = None
        self._scoring_tokenizer = None

    @property
    def perturbation_model(self):
        if self._perturbation_model is None:
            print(f"Loading perturbation model {self.perturbation_model_id} on {self.device}...")
            self._perturbation_model = AutoModelForSeq2SeqLM.from_pretrained(self.perturbation_model_id).to(self.device)
        return self._perturbation_model

    @property
    def perturbation_tokenizer(self):
        if self._perturbation_tokenizer is None:
            self._perturbation_tokenizer = AutoTokenizer.from_pretrained(self.perturbation_model_id)
        return self._perturbation_tokenizer

    @property
    def scoring_model(self):
        if self._scoring_model is None:
            print(f"Loading scoring model {self.scoring_model_id} on {self.device}...")
            self._scoring_model = GPT2LMHeadModel.from_pretrained(self.scoring_model_id).to(self.device)
            self._scoring_model.eval()
        return self._scoring_model

    @property
    def scoring_tokenizer(self):
        if self._scoring_tokenizer is None:
            self._scoring_tokenizer = GPT2TokenizerFast.from_pretrained(self.scoring_model_id)
        return self._scoring_tokenizer

    def _get_perplexity(self, text):
        """Calculates perplexity using the scoring model (similar to PerplexityCalculator)."""
        encodings = self.scoring_tokenizer(text, return_tensors='pt')
        seq_len = encodings.input_ids.size(1)
        max_length = self.scoring_model.config.n_positions
        stride = 512

        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.scoring_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
                num_valid_tokens = (target_ids != -100).sum().item()
                nll_sum += neg_log_likelihood * num_valid_tokens
                n_tokens += num_valid_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if n_tokens == 0:
            return float('nan')

        return torch.exp(nll_sum / n_tokens).item()

    def _perturb_text(self, text, span_length=3, pct=0.15, n_perturbations=1):
        """
        Generates perturbed versions of the text using T5.
        Masks random spans and asks T5 to fill them.
        """
        # Simple T5 span masking
        # T5 uses <extra_id_0>, <extra_id_1>, etc.

        tokens = text.split()
        n_tokens = len(tokens)
        n_spans = int(n_tokens * pct / span_length)

        perturbations = []

        for _ in range(n_perturbations):
            masked_tokens = tokens.copy()
            for i in range(n_spans):
                # Pick a random start
                start = np.random.randint(0, n_tokens - span_length)
                # Replace with a sentinel token
                sentinel = f"<extra_id_{i}>"
                masked_tokens[start:start+span_length] = [sentinel]
                # Note: T5 expects the sentinel in the input, and generates the text for it.
                # However, re-constructing the full text from T5 output is tricky.
                # A simpler approach for this task:
                # Use T5 to just "paraphrase" or "fill mask".

            # Actually, standard DetectGPT uses specific mask filling.
            # Let's simplify: split text into sentences, mask one word/span in a sentence, and fill it.
            # Or use a "paraphrase" prompt if the model supports it? T5 is good at "fill in the blank".

            # Let's try a simpler approach compatible with T5's pretraining objective.
            # Convert text to: "The <extra_id_0> sat on the mat." -> T5 generates "<extra_id_0> cat"

            # Construct the masked string properly handling multiple sentinels
            # T5 input format: "The <extra_id_0> sat on the <extra_id_1>."
            # We need to collapse consecutive masks?
            # Let's just do ONE perturbation pass with a few masks.

            # Simplified for robustness:
            # Just mask random 15% of words (or spans) with a SINGLE mask <extra_id_0> if we want to replace a chunk?
            # No, T5 can handle multiple.

            # Better implementation for robustness:
            # We will use a standard "mask-and-fill" loop.

            current_text = " ".join(masked_tokens)
            # Remove consecutive duplicates of the same sentinel?
            # Our logic above replaced span with ONE sentinel, but we need to ensure we don't overwrite sentinels.
            # This is getting complex for a "simplified" implementation.

            # Alternative: Use a dedicated mask-filling pipeline or just accept we are approximating.
            # Let's try to just mask ONE span of 10-15% length and let T5 fill it.

            span_len = int(len(tokens) * 0.15)
            start = np.random.randint(0, len(tokens) - span_len)
            prefix = " ".join(tokens[:start])
            suffix = " ".join(tokens[start+span_len:])
            masked_input = f"{prefix} <extra_id_0> {suffix}"

            input_ids = self.perturbation_tokenizer(masked_input, return_tensors="pt").input_ids.to(self.device)
            outputs = self.perturbation_model.generate(input_ids, max_length=50)
            fill_text = self.perturbation_tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Parse output to extract content for <extra_id_0>
            # Output usually looks like: "<pad> <extra_id_0> filled text <extra_id_1> ..."
            # We extract what's between <extra_id_0> and <extra_id_1> (or end)

            # Regex to extract
            match = re.search(r"<extra_id_0>(.*?)<extra_id_1>", fill_text)
            if not match:
                match = re.search(r"<extra_id_0>(.*?)(</s>|$)", fill_text)

            if match:
                filled_span = match.group(1).strip()
                new_text = f"{prefix} {filled_span} {suffix}"
                perturbations.append(new_text)
            else:
                perturbations.append(text) # Failed to perturb

        return perturbations

    def analyze(self, text, n_perturbations=5):
        """
        Calculates the surrogate perplexity score.
        Score = log(PPL_original) - mean(log(PPL_perturbed))

        Positive score -> Likely AI (Original is a local optimum)
        Negative/Zero score -> Likely Human
        """
        # Check Cache
        cached = self.cache.get(text, "surrogate_ppl")
        if cached:
            return cached

        # Get original PPL
        ppl_orig = self._get_perplexity(text)
        if np.isnan(ppl_orig):
             return {"surrogate_ppl_score": 0.0, "is_ai_likely": False}

        log_ppl_orig = np.log(ppl_orig)

        # Get perturbed PPLs
        perturbed_texts = self._perturb_text(text, n_perturbations=n_perturbations)
        log_ppls_perturbed = []

        for p_text in perturbed_texts:
            ppl = self._get_perplexity(p_text)
            if not np.isnan(ppl):
                log_ppls_perturbed.append(np.log(ppl))

        if not log_ppls_perturbed:
            result = {"surrogate_ppl_score": 0.0, "is_ai_likely": False}
            self.cache.set(text, "surrogate_ppl", result)
            return result

        mean_log_ppl_perturbed = np.mean(log_ppls_perturbed)

        # DetectGPT score
        # Note: In DetectGPT paper, curvature is log p(x) - E[log p(x_tilde)].
        # Higher probability = LOWER perplexity.
        # log p(x) is proportional to -log(PPL).
        # So: -log(PPL_orig) - E[-log(PPL_pert)]
        # = mean(log(PPL_pert)) - log(PPL_orig)

        score = mean_log_ppl_perturbed - log_ppl_orig

        # If score > 0, perturbed text has HIGHER perplexity (lower prob) than original.
        # This means original is a local maximum of probability (min of perplexity).
        # This suggests AI.

        result = {
            "surrogate_ppl_score": float(score),
            "original_ppl": float(ppl_orig),
            "perturbed_ppl_mean": float(np.exp(mean_log_ppl_perturbed)),
            "is_ai_likely": bool(score > 0) # Simple threshold
        }

        self.cache.set(text, "surrogate_ppl", result)
        return result
