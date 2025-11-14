# Revised Training Strategies Based on Loss Curve Analysis

## Key Insights from Training Curves

Your training curves show:
- **Training loss**: ~2.9 → 0.45 → 0.25 → 0.17 (decreasing)
- **Validation loss**: ~0.5 → 0.2 → 0.12 → 0.09 (decreasing, even below training)

**This means:**
- ✅ Model is **NOT overfitting** (validation loss still decreasing)
- ✅ Model **IS learning** (both losses decreasing)
- ❌ The broken SQL is likely from:
  1. **Too restrictive decoding parameters** (forcing weird outputs)
  2. **Encoder freezing** (preventing proper adaptation)
  3. **Under-training** (stopping too early)

---

## Step-by-Step Debugging Strategy

### Step 1: Simplify Decoding to See True Model Quality

**Problem:** Heavy decoding restrictions might be pushing the model into strange regions of the search space.

**Solution:** First, test with minimal decoding restrictions to see what the model "naturally" wants to output.

#### Option A: Minimal Decoding (Recommended First Test)

```bash
python train_t5.py \
  --finetune \
  --learning_rate 1e-4 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 10 \
  --patience_epochs 4 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 1 \
  --repetition_penalty 1.0 \
  --no_repeat_ngram_size 0 \
  --length_penalty 1.0 \
  --experiment_name t5_ft_simple_decode
```

**Key changes:**
- `--num_beams 1` (greedy decoding - simplest)
- `--repetition_penalty 1.0` (no penalty)
- `--no_repeat_ngram_size 0` (no blocking)
- `--length_penalty 1.0` (neutral)

**What to check:**
- Inspect a handful of dev predictions
- If SQL becomes much more syntactically correct → decoding was too restrictive
- If still broken → problem is in training, not decoding

#### Option B: Light Decoding (If Option A works)

```bash
python train_t5.py \
  --finetune \
  --learning_rate 1e-4 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 10 \
  --patience_epochs 4 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 4 \
  --repetition_penalty 1.0 \
  --no_repeat_ngram_size 0 \
  --length_penalty 0.8 \
  --experiment_name t5_ft_light_decode
```

**Key changes:**
- `--num_beams 4` (light beam search)
- `--repetition_penalty 1.0` (no penalty)
- `--no_repeat_ngram_size 0` (no blocking)
- `--length_penalty 0.8` (favor longer sequences)

---

### Step 2: Unfreeze Encoder (CRITICAL)

**Problem:** `--freeze_encoder` prevents the model from adapting to the SQL task.

**Solution:** Remove encoder freezing completely.

#### Option A: Full Fine-Tuning (Recommended)

```bash
python train_t5.py \
  --finetune \
  --learning_rate 1e-4 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 10 \
  --patience_epochs 4 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 4 \
  --repetition_penalty 1.0 \
  --no_repeat_ngram_size 0 \
  --length_penalty 0.8 \
  --experiment_name t5_ft_full_encoder
```

**Key change:** **NO `--freeze_encoder`** (encoder fully trainable)

**Why:** This is a specialized task (English → SQL). The encoder needs to learn task-specific representations.

#### Option B: Partially Frozen Encoder (If Memory Constrained)

```bash
python train_t5.py \
  --finetune \
  --freeze_n_encoder_layers 4 \
  --learning_rate 1e-4 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 10 \
  --patience_epochs 4 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 4 \
  --repetition_penalty 1.0 \
  --no_repeat_ngram_size 0 \
  --length_penalty 0.8 \
  --experiment_name t5_ft_partial_freeze
```

**Key change:** `--freeze_n_encoder_layers 4` (freeze only bottom 4 layers)

**Why:** Stabilizes training while still allowing higher layers to adapt.

---

### Step 3: Train Longer (Since Validation Loss Still Decreasing)

**Problem:** `--patience_epochs 3` might be stopping too early.

**Solution:** Increase patience to let training continue while validation loss is decreasing.

```bash
python train_t5.py \
  --finetune \
  --learning_rate 1e-4 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 15 \
  --patience_epochs 5 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 4 \
  --repetition_penalty 1.0 \
  --no_repeat_ngram_size 0 \
  --length_penalty 0.8 \
  --experiment_name t5_ft_longer
```

**Key changes:**
- `--max_n_epochs 15` (more epochs)
- `--patience_epochs 5` (more patience)

**Why:** Since validation loss is still decreasing, the model can benefit from more training.

---

### Step 4: Check Generation Limits

**Problem:** Broken SQL with incomplete clauses might be from `max_length` being too small.

**Current setting:** `max_length=512` (should be fine, but verify)

**Check in code:**
- `max_length` in `get_generation_kwargs()` is 512 (line 238)
- This should be sufficient for SQL queries
- If queries are getting truncated, increase to 256 or 512

**Verify:**
- Check if generated queries are being cut off mid-sentence
- If yes, increase `max_length` in generation kwargs

---

## Recommended Training Sequence

### Phase 1: Debug with Simple Decoding

**Goal:** See what the model naturally outputs without restrictions.

```bash
python train_t5.py \
  --finetune \
  --learning_rate 1e-4 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 10 \
  --patience_epochs 4 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 1 \
  --repetition_penalty 1.0 \
  --no_repeat_ngram_size 0 \
  --length_penalty 1.0 \
  --experiment_name t5_ft_debug_simple
```

**After this run:**
- Inspect dev predictions
- If SQL is much better → decoding was the problem
- If still broken → proceed to Phase 2

### Phase 2: Unfreeze Encoder

**Goal:** Let the model fully adapt to the SQL task.

```bash
python train_t5.py \
  --finetune \
  --learning_rate 1e-4 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 15 \
  --patience_epochs 5 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 4 \
  --repetition_penalty 1.0 \
  --no_repeat_ngram_size 0 \
  --length_penalty 0.8 \
  --experiment_name t5_ft_full_encoder
```

**Key:** NO `--freeze_encoder`!

**After this run:**
- Compare F1 scores to frozen-encoder run
- Should see significant improvement

### Phase 3: Fine-Tune Decoding (If Needed)

**Goal:** Add light restrictions only if repetition becomes an issue.

```bash
python train_t5.py \
  --finetune \
  --learning_rate 1e-4 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 15 \
  --patience_epochs 5 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 4 \
  --repetition_penalty 1.1 \
  --no_repeat_ngram_size 3 \
  --length_penalty 0.8 \
  --experiment_name t5_ft_final
```

**Key changes:**
- `--repetition_penalty 1.1` (light penalty, only if needed)
- `--no_repeat_ngram_size 3` (light blocking, only if needed)

**Only use this if:** You see repetition issues in Phase 2 outputs.

---

## Quick Reference: What Each Parameter Does

| Parameter | Value | Effect |
|-----------|-------|--------|
| `--num_beams` | 1 | Greedy (simplest) |
| `--num_beams` | 4 | Light beam search (balanced) |
| `--num_beams` | 5+ | Stronger search (slower) |
| `--repetition_penalty` | 1.0 | No penalty (natural output) |
| `--repetition_penalty` | 1.1-1.2 | Light penalty (if repetition) |
| `--repetition_penalty` | 1.3+ | Strong penalty (may cause issues) |
| `--no_repeat_ngram_size` | 0 | No blocking (natural) |
| `--no_repeat_ngram_size` | 3 | Light blocking (if repetition) |
| `--no_repeat_ngram_size` | 4+ | Strong blocking (may cause issues) |
| `--length_penalty` | 0.8 | Favor longer (good for SQL) |
| `--length_penalty` | 1.0 | Neutral |
| `--length_penalty` | 1.2+ | Favor shorter |

---

## Decision Tree

```
Start: Broken SQL outputs
│
├─ Step 1: Test with simple decoding (beams=1, no penalties)
│  │
│  ├─ If SQL becomes much better → Decoding was too restrictive
│  │  └─ Use simpler decoding settings
│  │
│  └─ If still broken → Proceed to Step 2
│
├─ Step 2: Unfreeze encoder (remove --freeze_encoder)
│  │
│  ├─ If F1 improves significantly → Encoder freezing was the problem
│  │  └─ Continue training without freezing
│  │
│  └─ If still issues → Proceed to Step 3
│
├─ Step 3: Train longer (more epochs, more patience)
│  │
│  └─ Since validation loss still decreasing, let it train more
│
└─ Step 4: Fine-tune decoding (only if repetition issues appear)
   │
   └─ Add light penalties only if needed
```

---

## Expected Results After Fixes

### Good Training Should Show:

```
Epoch 0: Train loss: 2.9, Dev loss: 0.5, F1: 0.15, Error: 45%
Epoch 1: Train loss: 0.45, Dev loss: 0.2, F1: 0.25, Error: 35%
Epoch 2: Train loss: 0.25, Dev loss: 0.12, F1: 0.35, Error: 25%
Epoch 3: Train loss: 0.17, Dev loss: 0.09, F1: 0.42, Error: 18%
Epoch 4: Train loss: 0.12, Dev loss: 0.08, F1: 0.48, Error: 15%
Epoch 5: Train loss: 0.10, Dev loss: 0.07, F1: 0.52, Error: 12%  ← Best
Epoch 6: Train loss: 0.09, Dev loss: 0.07, F1: 0.51, Error: 13%
→ Early stopping (no improvement for 5 epochs)
```

### Generated SQL Should Be:

- ✅ Syntactically valid (no `airport_service=1` errors)
- ✅ Proper table aliases (`airport_service_1`, not `airport_service-1`)
- ✅ Correct field types (time fields get numbers, not city names)
- ✅ Complete WHERE clauses
- ✅ Proper joins and conditions

---

## Final Recommended Command

**Start with this (combines all fixes):**

```bash
python train_t5.py \
  --finetune \
  --learning_rate 1e-4 \
  --batch_size 16 \
  --test_batch_size 16 \
  --max_n_epochs 15 \
  --patience_epochs 5 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 4 \
  --repetition_penalty 1.0 \
  --no_repeat_ngram_size 0 \
  --length_penalty 0.8 \
  --experiment_name t5_ft_revised
```

**Key points:**
- ✅ NO `--freeze_encoder`
- ✅ Simple decoding (no heavy penalties)
- ✅ More patience (5 epochs)
- ✅ Conservative learning rate (1e-4)

This should give you much better results!

---

## Troubleshooting Checklist

After each training run, check:

1. **Inspect dev predictions** - Are they syntactically valid?
2. **Compare F1 scores** - Is it improving?
3. **Check error rate** - Is it decreasing?
4. **Look at loss curves** - Are both decreasing?
5. **Sample outputs** - Do they make sense?

If still having issues, share:
- A few example queries (input, gold SQL, your SQL)
- Training loss curve
- Validation F1 progression

This will help diagnose whether remaining issues are "syntactic" or "logical".
