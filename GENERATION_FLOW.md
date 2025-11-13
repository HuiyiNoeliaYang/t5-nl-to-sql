# Generation Parameter Flow (Step-by-Step)

## How Parameters Are Applied During Generation

### Your Code → HuggingFace → Internal Implementation

```
┌─────────────────────────────────────────────────────────────────┐
│ Your Code (train_t5.py)                                         │
│                                                                  │
│  generation_kwargs = {                                          │
│      'repetition_penalty': 1.2,        ← You set this          │
│      'no_repeat_ngram_size': 3,        ← You set this          │
│      'length_penalty': 1.0,            ← You set this          │
│      ...                                                         │
│  }                                                               │
│                                                                  │
│  model.generate(input_ids, **generation_kwargs)                 │
│       ↓                                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ HuggingFace Transformers (transformers/generation/utils.py)     │
│                                                                  │
│  class GenerationMixin:                                         │
│      def generate(self, ...):                                   │
│          # Creates LogitsProcessor objects                      │
│          processors = [                                         │
│              RepetitionPenaltyLogitsProcessor(1.2),  ← Applied │
│              NoRepeatNGramLogitsProcessor(3),        ← Applied │
│          ]                                                      │
│          # Creates BeamScorer for beam search                   │
│          scorer = BeamSearchScorer(length_penalty=1.0)  ← Used │
│          # ... generation loop ...                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Generation Loop (At Each Step)                                  │
│                                                                  │
│  Step t: Generate token t                                       │
│                                                                  │
│  1. Model forward pass                                          │
│     logits = model(input_ids, decoder_input_ids)                │
│     # logits shape: [batch_size, vocab_size]                    │
│                                                                  │
│  2. Apply repetition_penalty (LogitsProcessor)                  │
│     for token_id in generated_sequence:                         │
│         if logits[token_id] > 0:                                │
│             logits[token_id] /= repetition_penalty              │
│         else:                                                    │
│             logits[token_id] *= repetition_penalty              │
│     # Reduces probability of tokens already seen                │
│                                                                  │
│  3. Apply no_repeat_ngram_size (LogitsProcessor)                │
│     last_n_minus_1 = generated_sequence[-(n-1):]                │
│     for token_id in range(vocab_size):                          │
│         potential_ngram = last_n_minus_1 + [token_id]           │
│         if potential_ngram in seen_ngrams:                      │
│             logits[token_id] = -inf  # Block completely         │
│     # Prevents duplicate n-grams                                │
│                                                                  │
│  4. Convert to probabilities                                    │
│     probs = softmax(logits / temperature)  # if sampling        │
│                                                                  │
│  5. Select next token                                           │
│     - Greedy: argmax(probs)                                     │
│     - Beam search: select top-k beams (uses length_penalty)     │
│     - Sampling: sample from probs                               │
│                                                                  │
│  6. Update generated_sequence and repeat                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Parameter Interaction

### Example: Generating SQL Query

```
Input: "what flights from denver to philadelphia"
Generated so far: "SELECT DISTINCT flight_1.flight_id FROM flight flight_1, airport_service airport_service_1, city city_1"

Step: Generate next token
```

#### Without Any Penalties:
```
Model logits:
  - "flight" (already seen): logit = 5.0 → prob = 0.15 (15%)
  - "WHERE": logit = 4.0 → prob = 0.12 (12%)
  - "AND": logit = 3.0 → prob = 0.09 (9%)
  ...
Result: Might generate "flight" again → repetition!
```

#### With repetition_penalty=1.2:
```
Model logits (before penalty):
  - "flight": logit = 5.0
  - "WHERE": logit = 4.0
  - "AND": logit = 3.0

After repetition_penalty=1.2:
  - "flight" (seen): logit = 5.0 / 1.2 = 4.17 → prob = 0.12 (12%) ↓
  - "WHERE": logit = 4.0 → prob = 0.14 (14%) ↑
  - "AND": logit = 3.0 → prob = 0.11 (11%)
  ...
Result: "WHERE" now more likely → reduces repetition
```

#### With no_repeat_ngram_size=3:
```
Last 2 tokens: ["city_1", ","]
Checking if next token would create duplicate 3-gram:

Candidate: "city_1" → ["city_1", ",", "city_1"] 
  - Already seen "city_1" in sequence? YES
  - Block: logit = -inf

Candidate: "airport_service_2" → ["city_1", ",", "airport_service_2"]
  - New 3-gram? YES
  - Allow: logit unchanged

Result: "city_1" blocked completely, must generate something new
```

#### With Both (repetition_penalty=1.2 + no_repeat_ngram_size=3):
```
1. Apply repetition_penalty: Reduce probability of all seen tokens
2. Apply no_repeat_ngram_size: Block tokens that create duplicate 3-grams
3. Select from remaining tokens

Result: Stronger prevention of repetition
```

#### With Beam Search + length_penalty=1.0:
```
Beam 1: "SELECT ... FROM flight" (score=-10.0, length=10)
  Normalized: -10.0 / 10^1.0 = -1.0

Beam 2: "SELECT ... FROM flight WHERE" (score=-12.0, length=12)
  Normalized: -12.0 / 12^1.0 = -1.0

Beam 3: "SELECT ... FROM flight WHERE city_1" (score=-15.0, length=15)
  Normalized: -15.0 / 15^1.0 = -1.0

All beams have same normalized score per token
→ Length penalty = 1.0 means "no length bias"
```

#### With Beam Search + length_penalty=0.8 (favor longer):
```
Beam 1: score=-10.0, length=10
  Normalized: -10.0 / 10^0.8 = -1.58

Beam 2: score=-12.0, length=12
  Normalized: -12.0 / 12^0.8 = -1.44  ← Better!

Beam 3: score=-15.0, length=15
  Normalized: -15.0 / 15^0.8 = -1.43  ← Best!

Result: Longer sequences favored (good for SQL queries)
```

## Key Takeaways

1. **repetition_penalty**: Applied at **every step** to all logits
   - Reduces (but doesn't eliminate) probability of repeated tokens
   - Works with any generation method

2. **no_repeat_ngram_size**: Applied at **every step** to block specific tokens
   - Completely blocks tokens that would create duplicate n-grams
   - More aggressive than repetition_penalty
   - Works with any generation method

3. **length_penalty**: Applied when **comparing beams** in beam search
   - Only affects beam search (not greedy, not sampling)
   - Normalizes beam scores by sequence length
   - Helps balance between shorter and longer sequences

## Recommended Settings for SQL Generation

```python
{
    'repetition_penalty': 1.2,        # Moderate penalty for repetition
    'no_repeat_ngram_size': 3,        # Block repeating 3-grams
    'length_penalty': 1.0,            # No length bias (or 0.8 to favor longer)
    'num_beams': 3,                   # Beam search for better quality
}
```

## When to Increase/Decrease

### Increase repetition_penalty (1.2 → 1.5):
- Still seeing repeated tokens/phrases
- Model is stuck in loops
- Output has duplicate WHERE clauses

### Increase no_repeat_ngram_size (3 → 4 or 5):
- Still seeing repeated patterns
- Longer n-grams are repeating
- Output has duplicate long phrases

### Adjust length_penalty:
- `length_penalty < 1.0` (e.g., 0.8): Favor longer sequences
  - Use when queries are too short
  - SQL queries need completeness
- `length_penalty > 1.0` (e.g., 1.2): Favor shorter sequences
  - Use when queries are too long
  - Want more concise outputs

