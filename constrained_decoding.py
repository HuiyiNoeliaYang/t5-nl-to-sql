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
        
        # Precompute all token IDs once (avoid encoding in hot path)
        self._select_token_ids = set(self.validator._token_cache.get('SELECT', set()))
        self._from_token_ids = set(self.validator._token_cache.get('FROM', set()))
        self._where_token_ids = set(self.validator._token_cache.get('WHERE', set()))
        self._close_paren_token_ids = set(self.validator._token_cache.get('CLOSE_PAREN', set()))
        
        # Precompute paren token IDs
        open_paren_ids = tokenizer.encode('(', add_special_tokens=False)
        self._open_paren_id = open_paren_ids[0] if open_paren_ids else -1
        
        # Cache will be populated when we know the device
        self._cached_tensors = {}
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Filter logits to only allow tokens that result in valid SQL.
        Optimized: Only checks structure using token IDs, avoids expensive decoding when possible.
        """
        batch_size = input_ids.shape[0]
        
        # Track if we've logged (to avoid spam)
        if not hasattr(self, '_logged'):
            print("🔒 Constrained decoding active: Filtering invalid SQL tokens")
            self._logged = True
        
        for batch_idx in range(batch_size):
            try:
                # Get current sequence (decoder input)
                current_ids = input_ids[batch_idx]
                
                # Fast tensor-based check: count occurrences without decoding
                non_pad_mask = (current_ids != self.tokenizer.pad_token_id) & (current_ids != self.tokenizer.eos_token_id)
                if not non_pad_mask.any():
                    continue
                
                filtered_ids = current_ids[non_pad_mask]
                
                # Cache tensors per device (create once, reuse)
                device_key = str(current_ids.device)
                if device_key not in self._cached_tensors:
                    self._cached_tensors[device_key] = {
                        'select': torch.tensor(list(self._select_token_ids), device=current_ids.device, dtype=current_ids.dtype),
                        'from': torch.tensor(list(self._from_token_ids), device=current_ids.device, dtype=current_ids.dtype),
                        'where': torch.tensor(list(self._where_token_ids), device=current_ids.device, dtype=current_ids.dtype),
                        'close_paren': torch.tensor(list(self._close_paren_token_ids), device=current_ids.device, dtype=current_ids.dtype),
                    }
                
                cached = self._cached_tensors[device_key]
                
                # Fast check: look for SELECT/FROM/WHERE token IDs directly (no decoding needed!)
                # Use cached tensors - much faster
                has_select = torch.any(torch.isin(filtered_ids, cached['select']))
                has_from = torch.any(torch.isin(filtered_ids, cached['from']))
                has_where = torch.any(torch.isin(filtered_ids, cached['where']))
                
                # Count parentheses using token IDs (no decoding needed)
                open_parens = (filtered_ids == self._open_paren_id).sum().item()
                if len(self._close_paren_token_ids) > 0:
                    close_parens = torch.sum(torch.isin(filtered_ids, cached['close_paren'])).item()
                else:
                    close_parens = 0
                
                # Apply constraints based on structure (NO DECODING - pure tensor ops)
                disallowed_tokens = set()
                
                # 1. Don't allow FROM before SELECT
                if not has_select and self._from_token_ids:
                    disallowed_tokens.update(self._from_token_ids)
                
                # 2. Don't allow WHERE before FROM
                if not has_from and self._where_token_ids:
                    disallowed_tokens.update(self._where_token_ids)
                
                # 3. Check balanced parentheses
                if close_parens >= open_parens and self._close_paren_token_ids:
                    disallowed_tokens.update(self._close_paren_token_ids)
                
                # Set scores for disallowed tokens to -inf (vectorized)
                if disallowed_tokens:
                    disallowed_tensor = torch.tensor(list(disallowed_tokens), device=scores.device, dtype=torch.long)
                    scores[batch_idx, disallowed_tensor] = float('-inf')
                        
            except Exception as e:
                # If validation fails, allow all tokens (fallback)
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

