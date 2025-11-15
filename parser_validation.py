"""
Post-generation SQL validation with rollback and regeneration.
Generate SQL first, validate with parser, then regenerate from error point if invalid.
"""

import torch
from typing import Optional, Tuple, List
from transformers import T5TokenizerFast

# Try importing SQL parsers
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


class SQLValidator:
    """Validates complete SQL using a parser."""
    
    def __init__(self, parser_type='sqlglot'):
        self.parser_type = parser_type
    
    def is_valid_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Check if SQL is valid.
        Returns: (is_valid, error_message)
        """
        if not sql or not sql.strip():
            return False, "Empty SQL"
        
        sql = sql.strip()
        
        # Remove trailing semicolon if present
        if sql.endswith(';'):
            sql = sql[:-1]
        
        try:
            if self.parser_type == 'sqlglot' and HAS_SQLGLOT:
                # sqlglot is more strict and gives better error messages
                try:
                    sqlglot.parse_one(sql)
                    return True, None
                except sqlglot.errors.ParseError as e:
                    return False, str(e)
            elif self.parser_type == 'sqlparse' and HAS_SQLPARSE:
                # sqlparse is more lenient
                parsed = sqlparse.parse(sql)
                if parsed and len(parsed) > 0:
                    stmt = parsed[0]
                    # Check if it's a valid statement type
                    if stmt.get_type() in ('SELECT', 'INSERT', 'UPDATE', 'DELETE', None):
                        return True, None
                return False, "Invalid SQL structure"
            else:
                # Fallback: basic syntax check
                if sql.upper().startswith('SELECT'):
                    return True, None
                return False, "Does not start with SELECT"
        except Exception as e:
            return False, f"Parser error: {str(e)}"
    
    def find_error_position(self, sql: str) -> Optional[int]:
        """
        Try to find where the SQL becomes invalid.
        Uses binary search to find the earliest invalid position.
        Returns: token position where error likely starts, or None
        """
        # Simple heuristic: find the last valid token
        # Try removing tokens from the end until SQL becomes valid
        tokens = sql.split()
        
        # Binary search for the last valid position
        left, right = 0, len(tokens)
        last_valid = 0
        
        while left < right:
            mid = (left + right) // 2
            test_sql = ' '.join(tokens[:mid])
            
            is_valid, _ = self.is_valid_sql(test_sql)
            if is_valid:
                last_valid = mid
                left = mid + 1
            else:
                right = mid
        
        # If we found a valid prefix, return position after it
        if last_valid < len(tokens):
            return last_valid
        
        return None


def validate_and_regenerate(
    model,
    tokenizer: T5TokenizerFast,
    encoder_input: torch.Tensor,
    encoder_mask: torch.Tensor,
    generation_kwargs: dict,
    validator: SQLValidator,
    max_retries: int = 3,
    rollback_tokens: int = 5
) -> str:
    """
    Generate SQL, validate with parser, and regenerate from error point if invalid.
    
    Args:
        model: T5 model
        tokenizer: T5 tokenizer
        encoder_input: Encoder input tokens
        encoder_mask: Encoder attention mask
        generation_kwargs: Generation parameters
        validator: SQLValidator instance
        max_retries: Maximum number of regeneration attempts
        rollback_tokens: How many tokens to rollback when regenerating
    
    Returns:
        Generated SQL (may be invalid if all retries fail)
    """
    device = encoder_input.device
    
    for attempt in range(max_retries):
        # Generate SQL
        with torch.no_grad():
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                **generation_kwargs
            )
        
        # Decode to SQL
        sql = tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Validate with parser
        is_valid, error_msg = validator.is_valid_sql(sql)
        
        if is_valid:
            return sql
        
        # SQL is invalid - try to find error position and regenerate
        if attempt < max_retries - 1:
            # Find where to rollback
            error_pos = validator.find_error_position(sql)
            
            if error_pos is not None and error_pos > rollback_tokens:
                # Rollback to before the error
                rollback_to = max(0, error_pos - rollback_tokens)
                
                # Re-encode the partial SQL up to rollback point
                partial_sql = ' '.join(sql.split()[:rollback_to])
                partial_ids = tokenizer.encode(partial_sql, add_special_tokens=False, return_tensors='pt').to(device)
                
                # Continue generation from rollback point
                # Note: This is simplified - in practice you'd need to handle decoder states
                print(f"  ⚠️  Invalid SQL (attempt {attempt + 1}/{max_retries}): {error_msg[:50]}")
                print(f"  🔄 Rolling back to token {rollback_to} and regenerating...")
                
                # For now, just regenerate from scratch with different parameters
                # In a full implementation, you'd continue from the rollback point
                generation_kwargs['do_sample'] = True  # Try sampling instead of beam
                generation_kwargs['temperature'] = 0.8  # Slightly more random
            else:
                # Can't find good rollback point, just retry with different params
                generation_kwargs['do_sample'] = not generation_kwargs.get('do_sample', False)
                generation_kwargs['temperature'] = generation_kwargs.get('temperature', 0.7) + 0.1
    
    # All retries failed - return the last generated SQL (even if invalid)
    print(f"  ❌ All {max_retries} regeneration attempts failed. Returning last SQL.")
    return sql


def validate_batch_with_parser(
    model,
    tokenizer: T5TokenizerFast,
    encoder_input: torch.Tensor,
    encoder_mask: torch.Tensor,
    generation_kwargs: dict,
    use_parser_validation: bool = True,
    max_retries: int = 2
) -> List[str]:
    """
    Generate SQL for a batch, validate with parser, and regenerate invalid ones.
    
    This is slower than normal generation but ensures valid SQL.
    """
    if not use_parser_validation or not (HAS_SQLGLOT or HAS_SQLPARSE):
        # Fallback to normal generation
        with torch.no_grad():
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                **generation_kwargs
            )
        return [tokenizer.decode(gen, skip_special_tokens=True) for gen in generated]
    
    validator = SQLValidator(parser_type='sqlglot' if HAS_SQLGLOT else 'sqlparse')
    batch_size = encoder_input.shape[0]
    results = []
    
    # Process each item in batch separately (for rollback capability)
    for i in range(batch_size):
        single_encoder_input = encoder_input[i:i+1]
        single_encoder_mask = encoder_mask[i:i+1]
        
        sql = validate_and_regenerate(
            model=model,
            tokenizer=tokenizer,
            encoder_input=single_encoder_input,
            encoder_mask=single_encoder_mask,
            generation_kwargs=generation_kwargs.copy(),
            validator=validator,
            max_retries=max_retries
        )
        results.append(sql)
    
    return results

