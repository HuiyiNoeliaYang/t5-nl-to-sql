"""
Constrained decoding for SQL generation.
Filters invalid tokens during generation to ensure valid SQL structure.
"""

import torch
from typing import Set
from transformers import LogitsProcessor
from transformers import T5TokenizerFast


class SQLGrammarValidator:
    """
    Validates partial SQL queries to filter invalid tokens during generation.
    Uses fast heuristics to block tokens that would break SQL structure.
    """
    
    def __init__(self, tokenizer=None):
        # Cache token IDs for performance (populated on first use)
        self._token_cache = {}
        if tokenizer is not None:
            self._precompute_tokens(tokenizer)
    
    def _precompute_tokens(self, tokenizer: T5TokenizerFast):
        """Precompute token IDs for SQL keywords to avoid repeated encoding."""
        if not self._token_cache:
            self._token_cache = {
                'SELECT': set(tokenizer.encode('SELECT', add_special_tokens=False)),
                'FROM': set(tokenizer.encode('FROM', add_special_tokens=False)),
                'WHERE': set(tokenizer.encode('WHERE', add_special_tokens=False)),
                'CLOSE_PAREN': set(tokenizer.encode(')', add_special_tokens=False)),
            }
    
    def get_disallowed_tokens(self, partial_text: str, tokenizer: T5TokenizerFast) -> Set[int]:
        """
        Get set of token IDs that should be BLOCKED (disallowed).
        Returns only the small set of tokens to filter, not all vocab.
        This is much more efficient than working with the full vocab set.
        """
        # Precompute tokens if not done
        if not self._token_cache:
            self._precompute_tokens(tokenizer)
        
        partial_upper = partial_text.upper().strip()
        
        # Fast path: Only return the small set of tokens to block
        disallowed_tokens = set()
        
        # 1. Don't allow FROM before SELECT (critical)
        if 'SELECT' not in partial_upper:
            disallowed_tokens.update(self._token_cache.get('FROM', set()))
        
        # 2. Don't allow WHERE before FROM (critical)
        if 'FROM' not in partial_upper:
            disallowed_tokens.update(self._token_cache.get('WHERE', set()))
        
        # 3. Check balanced parentheses (only if we have too many closing)
        open_parens = partial_text.count('(')
        close_parens = partial_text.count(')')
        if close_parens >= open_parens:
            disallowed_tokens.update(self._token_cache.get('CLOSE_PAREN', set()))
        
        return disallowed_tokens  # Return only the small set (typically 1-10 tokens)


class SQLConstrainedLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that filters tokens to ensure valid SQL generation.
    Uses SQLGrammarValidator to check if tokens are valid.
    """
    
    def __init__(self, tokenizer: T5TokenizerFast, validator: SQLGrammarValidator = None):
        self.tokenizer = tokenizer
        self.validator = validator or SQLGrammarValidator(tokenizer)
        self.vocab_size = len(tokenizer)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Filter logits to only allow tokens that result in valid SQL.
        
        Args:
            input_ids: Current input token IDs (batch_size, sequence_length)
            scores: Logits for next token (batch_size, vocab_size)
        
        Returns:
            Modified scores with invalid tokens set to -inf
        """
        batch_size = input_ids.shape[0]
        
        # Track if we've logged (to avoid spam)
        if not hasattr(self, '_logged'):
            print("🔒 Constrained decoding active: Filtering invalid SQL tokens")
            self._logged = True
        
        for batch_idx in range(batch_size):
            # Get current sequence (decoder input)
            current_ids = input_ids[batch_idx].tolist()
            
            # Decode to text (skip special tokens for validation)
            try:
                # Remove padding and special tokens for validation
                filtered_ids = [id for id in current_ids if id != self.tokenizer.pad_token_id 
                               and id != self.tokenizer.eos_token_id]
                
                if not filtered_ids:
                    continue
                
                partial_text = self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
                
                # Get disallowed tokens (fast version - only returns small set of blocked tokens)
                disallowed_tokens = self.validator.get_disallowed_tokens(partial_text, self.tokenizer)
                
                # Set scores for disallowed tokens to -inf (vectorized)
                # Only work with the small set of disallowed tokens, not all vocab
                if disallowed_tokens:
                    disallowed_tensor = torch.tensor(list(disallowed_tokens), device=scores.device, dtype=torch.long)
                    scores[batch_idx, disallowed_tensor] = float('-inf')
                        
            except Exception as e:
                # If validation fails, allow all tokens (fallback)
                # This prevents the decoder from getting stuck
                if not hasattr(self, '_error_logged'):
                    print(f"⚠️  Constrained decoding error (allowing all tokens): {e}")
                    self._error_logged = True
                pass
        
        return scores


def create_sql_constrained_processor(tokenizer: T5TokenizerFast, use_constrained: bool = True):
    """
    Factory function to create a constrained decoding processor.
    
    Args:
        tokenizer: T5 tokenizer
        use_constrained: Whether to enable constrained decoding
    
    Returns:
        SQLConstrainedLogitsProcessor or None
    """
    if not use_constrained:
        return None
    
    validator = SQLGrammarValidator()
    return SQLConstrainedLogitsProcessor(tokenizer, validator)

