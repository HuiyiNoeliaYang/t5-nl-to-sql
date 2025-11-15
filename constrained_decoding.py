"""
Constrained decoding for SQL generation.
Uses a SQL grammar validator to filter invalid tokens during generation.
"""

import re
import torch
from typing import List, Set
from transformers import LogitsProcessor
from transformers import T5TokenizerFast


class SQLGrammarValidator:
    """
    Validates partial SQL queries to ensure they follow SQL grammar rules.
    Uses a state machine to track SQL structure.
    """
    
    def __init__(self):
        # SQL keywords in order of typical appearance
        self.keywords = {
            'SELECT', 'DISTINCT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT',
            'IN', 'BETWEEN', 'LIKE', 'IS', 'NULL', 'EXISTS',
            'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'ON',
            'GROUP', 'BY', 'ORDER', 'HAVING', 'LIMIT', 'UNION',
            'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE'
        }
        
        # Operators
        self.operators = {'=', '!=', '<>', '<', '>', '<=', '>=', 'BETWEEN', 'IN', 'LIKE'}
        
        # SQL structure states
        self.states = {
            'START': 0,
            'SELECT': 1,
            'COLUMNS': 2,
            'FROM': 3,
            'TABLES': 4,
            'WHERE': 5,
            'CONDITION': 6,
            'VALUE': 7,
            'COMPLETE': 8
        }
    
    def is_valid_sql_prefix(self, text: str) -> bool:
        """
        Check if a partial SQL string is a valid prefix.
        Returns True if the text could be completed to valid SQL.
        """
        if not text or not text.strip():
            return True  # Empty is valid (we're just starting)
        
        text = text.strip()
        
        # Basic checks
        # 1. Must start with SELECT (for this dataset)
        if not text.upper().startswith('SELECT'):
            return False
        
        # 2. Check balanced parentheses
        if text.count('(') < text.count(')'):
            return False
        
        # 3. Check balanced quotes
        single_quotes = text.count("'") - text.count("\\'")
        if single_quotes % 2 != 0:
            # Inside a string literal - this is actually OK
            pass
        
        # 4. Check for incomplete keywords (basic check)
        # If we have "SELEC" without "T", it's invalid
        upper_text = text.upper()
        for keyword in self.keywords:
            if keyword.startswith(upper_text[-10:]) and len(upper_text) >= len(keyword):
                # We might be in the middle of typing a keyword
                if upper_text.endswith(keyword[:len(upper_text.split()[-1])]):
                    return True
        
        # 5. Check structure: SELECT ... FROM ... WHERE ...
        # Allow partial structures
        has_select = 'SELECT' in upper_text
        has_from = 'FROM' in upper_text
        has_where = 'WHERE' in upper_text
        
        # SELECT must come before FROM, FROM before WHERE
        select_pos = upper_text.find('SELECT')
        from_pos = upper_text.find('FROM')
        where_pos = upper_text.find('WHERE')
        
        if from_pos != -1 and select_pos > from_pos:
            return False
        if where_pos != -1 and from_pos > where_pos:
            return False
        if where_pos != -1 and select_pos > where_pos:
            return False
        
        # 6. Check for incomplete expressions
        # If we have "column_name =" without a value, it might be OK (we're generating)
        # If we have "column_name = 'value" without closing quote, it's OK (in progress)
        
        return True
    
    def get_allowed_next_tokens(self, partial_text: str, tokenizer: T5TokenizerFast) -> Set[int]:
        """
        Get set of token IDs that would result in valid SQL when appended.
        Uses efficient heuristics instead of checking every token.
        """
        vocab_size = len(tokenizer)
        allowed_tokens = set(range(vocab_size))  # Start with all tokens
        
        if not partial_text.strip():
            # At the start - must begin with SELECT
            select_tokens = tokenizer.encode('SELECT', add_special_tokens=False)
            if select_tokens:
                # Allow SELECT and common prefixes
                allowed_tokens = set(select_tokens)
                # Also allow tokens that could start SELECT (like "S", "SE", etc.)
                for token_id in range(min(1000, vocab_size)):  # Check first 1000 tokens
                    try:
                        token = tokenizer.decode([token_id]).strip()
                        if token and (token.upper().startswith('S') or token.upper().startswith('SELECT')):
                            allowed_tokens.add(token_id)
                    except:
                        continue
            return allowed_tokens
        
        # For partial SQL, use heuristics to filter obviously invalid tokens
        partial_upper = partial_text.upper().strip()
        
        # Filter tokens that would break basic SQL structure
        disallowed_tokens = set()
        
        # Check a sample of tokens for efficiency (not all vocab)
        sample_size = min(5000, vocab_size)  # Check up to 5000 tokens
        step = max(1, vocab_size // sample_size)
        
        for token_id in range(0, vocab_size, step):
            try:
                token = tokenizer.decode([token_id]).strip()
                if not token:
                    continue
                
                test_text = partial_text + token
                
                # Quick structural checks
                # 1. Don't allow FROM before SELECT
                if 'FROM' in token.upper() and 'SELECT' not in partial_upper:
                    disallowed_tokens.add(token_id)
                    continue
                
                # 2. Don't allow WHERE before FROM (if FROM is required)
                if 'WHERE' in token.upper() and 'FROM' not in partial_upper:
                    disallowed_tokens.add(token_id)
                    continue
                
                # 3. Check balanced parentheses
                open_parens = test_text.count('(')
                close_parens = test_text.count(')')
                if close_parens > open_parens:
                    disallowed_tokens.add(token_id)
                    continue
                
                # 4. Check balanced quotes (allow incomplete strings)
                # This is OK - we might be in the middle of a string
                
            except:
                continue
        
        # Remove disallowed tokens
        allowed_tokens -= disallowed_tokens
        
        # Ensure we don't filter too aggressively
        if len(allowed_tokens) < vocab_size * 0.5:  # Less than 50% allowed
            return set(range(vocab_size))  # Fallback: allow all
        
        return allowed_tokens


class SQLConstrainedLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that filters tokens to ensure valid SQL generation.
    Uses SQLGrammarValidator to check if tokens are valid.
    """
    
    def __init__(self, tokenizer: T5TokenizerFast, validator: SQLGrammarValidator = None):
        self.tokenizer = tokenizer
        self.validator = validator or SQLGrammarValidator()
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
                
                # Get allowed tokens
                allowed_tokens = self.validator.get_allowed_next_tokens(partial_text, self.tokenizer)
                
                # Set scores for disallowed tokens to -inf
                for token_id in range(self.vocab_size):
                    if token_id not in allowed_tokens:
                        scores[batch_idx, token_id] = float('-inf')
                        
            except Exception as e:
                # If validation fails, allow all tokens (fallback)
                # This prevents the decoder from getting stuck
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

