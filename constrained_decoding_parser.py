"""
Alternative constrained decoding using an actual SQL parser.
This shows how to use sqlparse or sqlglot for more accurate validation.
"""

import torch
from typing import Set
from transformers import LogitsProcessor, T5TokenizerFast

# Try importing SQL parsers (install with: pip install sqlparse sqlglot)
try:
    import sqlparse
    HAS_SQLPARSE = True
except ImportError:
    HAS_SQLPARSE = False
    print("⚠️  sqlparse not installed. Install with: pip install sqlparse")

try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False
    print("⚠️  sqlglot not installed. Install with: pip install sqlglot")


class ParserBasedSQLValidator:
    """
    Uses an actual SQL parser to validate partial SQL.
    More accurate than heuristics, but has challenges (see below).
    """
    
    def __init__(self, parser_type='sqlparse'):
        self.parser_type = parser_type
        self._token_cache = {}
    
    def _precompute_tokens(self, tokenizer: T5TokenizerFast):
        """Precompute token IDs for SQL keywords."""
        if not self._token_cache:
            self._token_cache = {
                'FROM': set(tokenizer.encode('FROM', add_special_tokens=False)),
                'WHERE': set(tokenizer.encode('WHERE', add_special_tokens=False)),
                'CLOSE_PAREN': set(tokenizer.encode(')', add_special_tokens=False)),
            }
    
    def _is_valid_sql_prefix(self, partial_sql: str) -> bool:
        """
        Check if partial SQL can be completed to valid SQL.
        
        CHALLENGE: Most parsers expect complete SQL, not partial!
        We need to work around this.
        """
        if not partial_sql.strip():
            return True
        
        # Strategy 1: Try to parse as-is (might fail for incomplete SQL)
        try:
            if self.parser_type == 'sqlparse' and HAS_SQLPARSE:
                # sqlparse can handle some incomplete SQL
                parsed = sqlparse.parse(partial_sql)
                if parsed:
                    # Check if it's a SELECT statement
                    stmt = parsed[0]
                    return stmt.get_type() == 'SELECT' or stmt.get_type() is None
            elif self.parser_type == 'sqlglot' and HAS_SQLGLOT:
                # sqlglot is stricter - will fail on incomplete SQL
                try:
                    sqlglot.parse_one(partial_sql)
                    return True
                except:
                    # Incomplete SQL - try to see if we can complete it
                    # Add a dummy completion and see if it parses
                    test_sql = partial_sql + " FROM dummy_table"
                    try:
                        sqlglot.parse_one(test_sql)
                        return True
                    except:
                        return False
        except Exception:
            pass
        
        # Strategy 2: Heuristic fallback for incomplete SQL
        # If parser fails, use simple checks
        partial_upper = partial_sql.upper()
        
        # Must start with SELECT
        if not partial_upper.strip().startswith('SELECT'):
            return False
        
        # Check balanced parentheses
        if partial_sql.count('(') < partial_sql.count(')'):
            return False
        
        # Check basic structure
        select_pos = partial_upper.find('SELECT')
        from_pos = partial_upper.find('FROM')
        where_pos = partial_upper.find('WHERE')
        
        if from_pos != -1 and select_pos > from_pos:
            return False
        if where_pos != -1 and from_pos > where_pos:
            return False
        
        return True
    
    def get_disallowed_tokens(self, partial_text: str, tokenizer: T5TokenizerFast) -> Set[int]:
        """
        Get tokens to block using SQL parser validation.
        
        CHALLENGE: We need to test each possible token, which is slow!
        """
        self._precompute_tokens(tokenizer)
        
        if not partial_text.strip():
            # At start - must be SELECT
            return set()  # Allow all initially, or filter non-SELECT starts
        
        disallowed_tokens = set()
        partial_upper = partial_text.upper()
        
        # Fast checks first (same as before)
        if 'SELECT' not in partial_upper:
            disallowed_tokens.update(self._token_cache.get('FROM', set()))
        if 'FROM' not in partial_upper:
            disallowed_tokens.update(self._token_cache.get('WHERE', set()))
        
        # PROBLEM: To use parser properly, we'd need to test each token:
        # for token_id in range(vocab_size):
        #     token = tokenizer.decode([token_id])
        #     test_sql = partial_text + token
        #     if not self._is_valid_sql_prefix(test_sql):
        #         disallowed_tokens.add(token_id)
        # 
        # This is TOO SLOW! (32,000 token decodes per step)
        
        # So we only use parser for critical tokens
        # Check if adding common SQL keywords would break it
        critical_tokens = ['FROM', 'WHERE', 'SELECT', 'AND', 'OR', ')', ';']
        for keyword in critical_tokens:
            keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
            for token_id in keyword_tokens:
                test_sql = partial_text + tokenizer.decode([token_id])
                if not self._is_valid_sql_prefix(test_sql):
                    disallowed_tokens.add(token_id)
        
        return disallowed_tokens


class ParserBasedConstrainedProcessor(LogitsProcessor):
    """
    Constrained decoder using actual SQL parser.
    More accurate but slower than heuristics.
    """
    
    def __init__(self, tokenizer: T5TokenizerFast, parser_type='sqlparse'):
        self.tokenizer = tokenizer
        self.validator = ParserBasedSQLValidator(parser_type)
        self.vocab_size = len(tokenizer)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            current_ids = input_ids[batch_idx].tolist()
            filtered_ids = [id for id in current_ids 
                          if id != self.tokenizer.pad_token_id 
                          and id != self.tokenizer.eos_token_id]
            
            if not filtered_ids:
                continue
            
            try:
                partial_text = self.tokenizer.decode(filtered_ids, skip_special_tokens=True)
                disallowed_tokens = self.validator.get_disallowed_tokens(partial_text, self.tokenizer)
                
                if disallowed_tokens:
                    disallowed_tensor = torch.tensor(
                        list(disallowed_tokens), 
                        device=scores.device, 
                        dtype=torch.long
                    )
                    scores[batch_idx, disallowed_tensor] = float('-inf')
            except Exception:
                pass
        
        return scores

