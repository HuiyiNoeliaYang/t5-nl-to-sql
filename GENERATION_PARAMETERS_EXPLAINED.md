# Generation Parameters Implementation Explained

## Overview

In your code, `repetition_penalty`, `no_repeat_ngram_size`, and `length_penalty` are **not implemented directly** - they're passed as parameters to HuggingFace Transformers' `model.generate()` method, which implements them internally.

## How They're Used in Your Code

### 1. Parameter Collection (`get_generation_kwargs`)

```python
def get_generation_kwargs(args, tokenizer):
    base_kwargs = {
        'max_length': 512,
        'repetition_penalty': args.repetition_penalty,      # ← Passed to model.generate()
        'no_repeat_ngram_size': args.no_repeat_ngram_size,  # ← Passed to model.generate()
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    # ... beam search or sampling config ...
    return base_kwargs
```

### 2. Generation Call

```python
generation_kwargs = get_generation_kwargs(args, tokenizer)
generated = model.generate(
    input_ids=encoder_input,
    attention_mask=encoder_mask,
    **generation_kwargs  # ← All parameters unpacked here
)
```

The `model` is a `T5ForConditionalGeneration` instance from HuggingFace Transformers, which inherits `generate()` from `GenerationMixin`.

---

## How They're Implemented in HuggingFace Transformers

### 1. `repetition_penalty` (Repetition Penalty)

**What it does:**
- Penalizes tokens that have already appeared in the generated sequence
- Reduces the probability of repeating tokens/patterns

**Algorithm (simplified):**
```python
# Pseudocode of how HuggingFace implements it
def apply_repetition_penalty(logits, input_ids, penalty):
    """
    logits: [vocab_size] - raw model output logits
    input_ids: [seq_len] - tokens already generated
    penalty: float - penalty factor (1.0 = no penalty, >1.0 = penalty)
    """
    # Create a set of tokens that have appeared
    seen_tokens = set(input_ids)
    
    # Apply penalty to logits of seen tokens
    for token_id in seen_tokens:
        if logits[token_id] > 0:
            logits[token_id] = logits[token_id] / penalty  # Divide by penalty (reduces probability)
        else:
            logits[token_id] = logits[token_id] * penalty  # Multiply by penalty (if negative)
    
    return logits
```

**Where it's applied:**
- Applied **at each generation step** before computing probabilities
- Only penalizes tokens that have appeared in the **current generated sequence** (not the input)

**Effect:**
- `repetition_penalty = 1.0`: No penalty (default behavior)
- `repetition_penalty = 1.2`: 20% penalty (logits divided by 1.2 for repeated tokens)
- `repetition_penalty = 2.0`: 50% penalty (stronger suppression of repeats)

**Example:**
```
Generated so far: "SELECT DISTINCT flight_1.flight_id FROM flight"
Next token probabilities:
  - "flight" (seen): logit = 5.0 → after penalty (1.2): logit = 4.17
  - "WHERE" (not seen): logit = 4.0 → unchanged
  - "AND" (not seen): logit = 3.0 → unchanged
```

---

### 2. `no_repeat_ngram_size` (N-gram Blocking)

**What it does:**
- Completely **blocks** n-grams (sequences of n tokens) from repeating
- More aggressive than repetition_penalty - sets probability to negative infinity

**Algorithm (simplified):**
```python
# Pseudocode of how HuggingFace implements it
def apply_no_repeat_ngram(logits, generated_sequence, ngram_size):
    """
    logits: [vocab_size] - raw model output logits
    generated_sequence: [seq_len] - tokens already generated
    ngram_size: int - size of n-gram to block (e.g., 3 = trigrams)
    """
    # Extract all n-grams from generated sequence
    generated_ngrams = []
    for i in range(len(generated_sequence) - ngram_size + 1):
        ngram = tuple(generated_sequence[i:i+ngram_size])
        generated_ngrams.append(ngram)
    
    # For each possible next token
    for token_id in range(vocab_size):
        # Check if adding this token would create a duplicate n-gram
        potential_ngram = tuple(generated_sequence[-(ngram_size-1):] + [token_id])
        
        if potential_ngram in generated_ngrams:
            # Block this token completely
            logits[token_id] = float('-inf')  # Set to negative infinity
    
    return logits
```

**Where it's applied:**
- Applied **at each generation step** before sampling/beam search
- Checks if adding the next token would create a duplicate n-gram

**Effect:**
- `no_repeat_ngram_size = 3`: Prevents any 3-token sequence from repeating
- `no_repeat_ngram_size = 4`: Prevents any 4-token sequence from repeating
- `no_repeat_ngram_size = 0`: No blocking (default)

**Example:**
```
Generated so far: "flight_1.to_airport = airport_service_2.airport_code AND"
Last 2 tokens: ["airport_code", "AND"]
no_repeat_ngram_size = 3

If model wants to generate "airport_service_2.airport_code AND" again:
  - Check: ["airport_code", "AND", "airport_service_2"] already seen?
  - If yes: Set logit to -inf (completely blocked)
  - Forces model to generate something different
```

---

### 3. `length_penalty` (Length Penalty - Beam Search Only)

**What it does:**
- Adjusts beam search scores based on sequence length
- Prevents beam search from favoring shorter sequences
- Only applies to **beam search**, not greedy decoding or sampling

**Algorithm (simplified):**
```python
# Pseudocode of how HuggingFace implements it
def apply_length_penalty(beam_score, sequence_length, length_penalty):
    """
    beam_score: float - cumulative log probability of the beam
    sequence_length: int - current length of the sequence
    length_penalty: float - penalty factor
    """
    # Normalize score by length
    # Formula: score = beam_score / (sequence_length ** length_penalty)
    normalized_score = beam_score / (sequence_length ** length_penalty)
    return normalized_score
```

**Where it's applied:**
- Applied when **comparing beams** during beam search
- Used to rank beams before selecting top-k for next step

**Effect:**
- `length_penalty = 1.0`: No penalty (scores divided by length^1.0 = length)
- `length_penalty < 1.0`: Favors **longer** sequences (e.g., 0.8)
- `length_penalty > 1.0`: Favors **shorter** sequences (e.g., 1.2)

**Example:**
```
Beam 1: Score = -10.0, Length = 20 tokens
  Normalized (penalty=1.0): -10.0 / 20^1.0 = -0.5

Beam 2: Score = -12.0, Length = 15 tokens  
  Normalized (penalty=1.0): -12.0 / 15^1.0 = -0.8

Beam 1 wins (higher normalized score despite lower raw score)

With length_penalty = 0.8 (favor longer):
  Beam 1: -10.0 / 20^0.8 = -0.87
  Beam 2: -12.0 / 15^0.8 = -1.23
  Beam 1 wins even more strongly
```

---

## Execution Order During Generation

When `model.generate()` is called, here's the order of operations at **each step**:

```
1. Model forward pass → Get logits [vocab_size]

2. Apply repetition_penalty:
   - Reduce logits of tokens already in generated_sequence
   - logits[seen_token] = logits[seen_token] / repetition_penalty

3. Apply no_repeat_ngram_size:
   - Check if next token would create duplicate n-gram
   - Set logits[blocked_token] = -inf

4. Apply temperature (if sampling):
   - logits = logits / temperature

5. Apply top_k / top_p (if sampling):
   - Filter to top-k tokens or nucleus

6. Convert to probabilities:
   - probs = softmax(logits)

7. Sample or select (greedy/beam search):
   - Greedy: argmax(probs)
   - Beam search: select top-k beams (using length_penalty for ranking)
   - Sampling: sample from probs

8. Add token to sequence and repeat
```

---

## Key Differences

| Parameter | When Applied | Effect | Applies To |
|-----------|--------------|--------|------------|
| `repetition_penalty` | Every step | Reduces probability of repeated tokens | All methods |
| `no_repeat_ngram_size` | Every step | Blocks duplicate n-grams completely | All methods |
| `length_penalty` | Beam ranking | Normalizes beam scores by length | Beam search only |

---

## Implementation Location in HuggingFace

These are implemented in:
- **File**: `transformers/generation/utils.py`
- **Classes**: `LogitsProcessor` (repetition_penalty, no_repeat_ngram_size)
- **Classes**: `BeamScorer` (length_penalty for beam search)

The actual code is complex and handles edge cases, batching, device management, etc., but the core logic is as described above.

---

## Debugging Tips

To see what's happening, you can:

1. **Check generated sequences** for repeated patterns
2. **Adjust parameters** incrementally:
   - Start with `repetition_penalty = 1.2`
   - Add `no_repeat_ngram_size = 3`
   - Increase gradually if issues persist
3. **Compare outputs** with/without these parameters
4. **Monitor** if model is getting stuck in loops

---

## References

- HuggingFace Transformers Documentation: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
- Source code: `transformers/generation/utils.py`
- Paper: "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019) - discusses repetition issues

