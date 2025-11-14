# Training Guide: How to Train Your T5 Model

## Overview

After implementing the generation parameter improvements, here's how to train your model effectively. The key issues to address:

1. **Learning rate is too high** (default 0.1 is ~1000x too high for T5 fine-tuning)
2. **Need proper training schedule** (epochs, patience, warmup)
3. **Monitor both loss and F1** (low loss doesn't mean good F1)
4. **Use improved generation parameters** (beam search, repetition penalty)

---

## Recommended Training Commands

### Option 1: Fine-tuning (Recommended - Start Here)

```bash
python train_t5.py \
  --finetune \
  --learning_rate 3e-4 \
  --batch_size 16 \
  --max_n_epochs 10 \
  --patience_epochs 3 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 5 \
  --repetition_penalty 1.3 \
  --no_repeat_ngram_size 4 \
  --length_penalty 1.0 \
  --experiment_name t5_ft_improved
```

**Why these parameters:**
- `--learning_rate 3e-4`: Standard for T5 fine-tuning (NOT 0.1!)
- `--num_beams 5`: Better quality than greedy
- `--repetition_penalty 1.3`: Stronger penalty for repetition
- `--no_repeat_ngram_size 4`: Block repeating 4-grams
- `--patience_epochs 3`: Stop if F1 doesn't improve for 3 epochs
- `--weight_decay 0.01`: Light regularization

### Option 2: Fine-tuning with Layer Freezing (If Limited GPU Memory)

```bash
python train_t5.py \
  --finetune \
  --freeze_n_encoder_layers 4 \
  --learning_rate 5e-4 \
  --batch_size 16 \
  --max_n_epochs 15 \
  --patience_epochs 3 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 5 \
  --repetition_penalty 1.3 \
  --no_repeat_ngram_size 4 \
  --length_penalty 1.0 \
  --experiment_name t5_ft_freeze_4_layers
```

**Why freeze layers:**
- Reduces memory usage
- Faster training
- Often works well for fine-tuning (lower layers learn task-specific patterns)

### Option 3: Training from Scratch (Not Recommended, But Available)

```bash
python train_t5.py \
  --learning_rate 1e-3 \
  --batch_size 16 \
  --max_n_epochs 20 \
  --patience_epochs 5 \
  --scheduler_type cosine \
  --num_warmup_epochs 2 \
  --weight_decay 0.01 \
  --num_beams 5 \
  --repetition_penalty 1.3 \
  --no_repeat_ngram_size 4 \
  --length_penalty 1.0 \
  --experiment_name t5_scratch
```

**Note:** Training from scratch requires much more data and time. Fine-tuning is strongly recommended.

---

## Step-by-Step Training Process

### Step 1: Prepare Your Environment

```bash
# Activate your environment
conda activate hw4-part-2-nlp  # or your env name

# Make sure you have the data
ls data/train.nl data/train.sql data/dev.nl data/dev.sql

# Check GPU availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Start Training with Recommended Parameters

```bash
# Start training (this will take a while)
python train_t5.py \
  --finetune \
  --learning_rate 3e-4 \
  --batch_size 16 \
  --max_n_epochs 10 \
  --patience_epochs 3 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 5 \
  --repetition_penalty 1.3 \
  --no_repeat_ngram_size 4 \
  --length_penalty 1.0 \
  --experiment_name t5_ft_v1
```

### Step 3: Monitor Training

Watch for:
- **Training loss**: Should decrease steadily
- **Validation loss**: Should decrease and track training loss
- **Record F1**: **Most important metric** - should increase
- **SQL EM**: Exact match on SQL queries
- **Error rate**: Should decrease (fewer SQL syntax errors)

Expected output:
```
Epoch 0: Average train loss was 2.3456
Epoch 0: Dev loss: 2.1234, Record F1: 0.1234, Record EM: 0.0567, SQL EM: 0.0234
Epoch 0: 45.23% of the generated outputs led to SQL errors

Epoch 1: Average train loss was 1.8765
Epoch 1: Dev loss: 1.6543, Record F1: 0.2345, Record EM: 0.1234, SQL EM: 0.0567
Epoch 1: 32.15% of the generated outputs led to SQL errors

...
```

### Step 4: Check Training Curves

After each epoch, training curves are saved to:
```
checkpoints/ft_experiments/t5_ft_v1/training_curves.png
```

**What to look for:**
- Training loss should decrease
- Validation loss should decrease
- **Validation F1 should increase** (this is key!)
- If validation F1 plateaus or decreases, training may be overfitting

### Step 5: Evaluate Best Model

After training completes, the best model (based on validation F1) is automatically loaded and evaluated on dev and test sets.

Results are saved to:
- Dev SQL: `results/t5_ft_t5_ft_v1_dev.sql`
- Dev Records: `records/t5_ft_t5_ft_v1_dev.pkl`
- Test SQL: `results/t5_ft_t5_ft_v1_test.sql`
- Test Records: `records/t5_ft_t5_ft_v1_test.pkl`

### Step 6: Evaluate F1 Score

```bash
python evaluate.py \
  --predicted_sql results/t5_ft_t5_ft_v1_dev.sql \
  --predicted_records records/t5_ft_t5_ft_v1_dev.pkl \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl
```

---

## Hyperparameter Tuning Guide

### Learning Rate (CRITICAL!)

**Problem:** Default is 0.1, which is ~1000x too high for T5 fine-tuning!

**Recommended:**
- Fine-tuning: `3e-4` to `5e-4` (0.0003 to 0.0005)
- From scratch: `1e-3` to `3e-3` (0.001 to 0.003)

**How to choose:**
- Start with `3e-4` for fine-tuning
- If training is too slow, try `5e-4`
- If training is unstable (loss spikes), try `1e-4`
- **Never use 0.1** unless training from scratch with millions of examples!

### Batch Size

**Recommended:**
- GPU memory allows: `16` to `32`
- Limited GPU memory: `8` to `16`
- Very limited: `4` to `8`

**Trade-offs:**
- Larger batch size: More stable training, faster (per epoch), but slower per iteration
- Smaller batch size: More iterations per epoch, can help generalization

### Number of Epochs

**Recommended:**
- Fine-tuning: `10` to `20` epochs
- From scratch: `20` to `50` epochs

**With early stopping:**
- Set `--patience_epochs 3` to stop if F1 doesn't improve for 3 epochs
- Prevents overfitting

### Generation Parameters

**Beam Search:**
- `--num_beams 3` to `5`: Good balance of quality and speed
- `--num_beams 10`: Better quality, but slower

**Repetition Control:**
- `--repetition_penalty 1.2`: Moderate penalty
- `--repetition_penalty 1.3` to `1.5`: Stronger penalty (if still repeating)
- `--no_repeat_ngram_size 3` to `4`: Block repeating n-grams

**Length Penalty:**
- `--length_penalty 1.0`: No bias (default)
- `--length_penalty 0.8`: Favor longer sequences (good for SQL)
- `--length_penalty 1.2`: Favor shorter sequences

---

## Troubleshooting

### Issue 1: Loss is Very High (> 5.0)

**Causes:**
- Learning rate too high (check if using 0.1!)
- Model not initialized properly
- Data issues

**Solutions:**
- Use `--learning_rate 3e-4` (not 0.1!)
- Check data loading (are queries tokenized correctly?)
- Verify model is loaded correctly

### Issue 2: Loss Decreases But F1 Doesn't Improve

**Causes:**
- Model learning wrong patterns
- Generation parameters not optimal
- Training data issues

**Solutions:**
- Check generation parameters (use beam search, repetition penalty)
- Inspect generated outputs (are they malformed?)
- Try different learning rates
- Check if training data is correct

### Issue 3: Model Generates Repetitive Output

**Causes:**
- Repetition penalty too low
- No n-gram blocking
- Model stuck in loops

**Solutions:**
- Increase `--repetition_penalty 1.3` to `1.5`
- Increase `--no_repeat_ngram_size 4` to `5`
- Use beam search instead of greedy
- Check if model is overfitting

### Issue 4: Model Generates Wrong City Names

**Causes:**
- Model not learning NL → SQL mapping
- Training data issues
- Learning rate too high (causing unstable training)

**Solutions:**
- Lower learning rate (try `1e-4` to `3e-4`)
- Train for more epochs
- Check training data quality
- Verify city names in training data match SQL

### Issue 5: Training is Too Slow

**Solutions:**
- Reduce batch size (if memory allows, increase instead)
- Use GPU (if not already)
- Freeze more layers (`--freeze_n_encoder_layers 4`)
- Reduce number of beams during training (but use more for evaluation)

### Issue 6: Out of Memory (OOM)

**Solutions:**
- Reduce batch size (`--batch_size 8` or `4`)
- Freeze layers (`--freeze_n_encoder_layers 4`)
- Use gradient accumulation (if implemented)
- Reduce max_length during generation

---

## Expected Results

### Fine-tuning (Good Setup)

After 5-10 epochs with proper hyperparameters:

- **Training loss**: Should decrease from ~3.0 to ~0.3-0.5
- **Validation loss**: Should track training loss closely
- **Record F1**: Should reach 0.3-0.6 (depending on data quality)
- **SQL EM**: Should reach 0.1-0.3 (exact match is hard)
- **Error rate**: Should decrease from ~50% to ~10-20%

### What Good Training Looks Like

```
Epoch 0: Train loss: 2.5, Dev loss: 2.3, F1: 0.15, Error: 45%
Epoch 1: Train loss: 1.8, Dev loss: 1.7, F1: 0.25, Error: 35%
Epoch 2: Train loss: 1.2, Dev loss: 1.1, F1: 0.35, Error: 25%
Epoch 3: Train loss: 0.8, Dev loss: 0.7, F1: 0.42, Error: 18%
Epoch 4: Train loss: 0.5, Dev loss: 0.6, F1: 0.48, Error: 15%
Epoch 5: Train loss: 0.4, Dev loss: 0.5, F1: 0.52, Error: 12%  ← Best model
Epoch 6: Train loss: 0.3, Dev loss: 0.6, F1: 0.51, Error: 13%  ← Overfitting?
```

**Key observations:**
- Loss decreases steadily
- F1 increases
- Error rate decreases
- Dev loss starts increasing (overfitting) while F1 plateaus

---

## Advanced: Hyperparameter Search

### Quick Search: Learning Rate

Try different learning rates:
```bash
# Low learning rate
python train_t5.py --finetune --learning_rate 1e-4 --experiment_name lr_1e4 ...

# Medium learning rate (recommended)
python train_t5.py --finetune --learning_rate 3e-4 --experiment_name lr_3e4 ...

# Higher learning rate
python train_t5.py --finetune --learning_rate 5e-4 --experiment_name lr_5e4 ...
```

Compare F1 scores and choose the best.

### Quick Search: Generation Parameters

Try different generation settings:
```bash
# Conservative (fewer repetitions)
python train_t5.py ... --repetition_penalty 1.5 --no_repeat_ngram_size 5 ...

# Moderate (recommended)
python train_t5.py ... --repetition_penalty 1.3 --no_repeat_ngram_size 4 ...

# Light (allows some repetition)
python train_t5.py ... --repetition_penalty 1.1 --no_repeat_ngram_size 3 ...
```

---

## Monitoring and Logging

### Using WandB (Optional but Recommended)

```bash
python train_t5.py \
  --finetune \
  --use_wandb \
  --learning_rate 3e-4 \
  ... \
  --experiment_name t5_ft_wandb
```

This will log:
- Training/validation loss
- F1 scores
- SQL EM
- Error rates
- Learning rate schedule

### Manual Monitoring

Check training curves:
```bash
# View training curves (updated after each epoch)
open checkpoints/ft_experiments/t5_ft_v1/training_curves.png
```

Check outputs:
```bash
# Compare generated SQL with ground truth
head -20 results/t5_ft_t5_ft_v1_dev.sql
head -20 data/dev.sql
```

---

## Final Checklist

Before training:
- [ ] Learning rate is set correctly (3e-4 for fine-tuning, NOT 0.1!)
- [ ] Generation parameters are set (beam search, repetition penalty)
- [ ] Data is loaded correctly (check train/dev files exist)
- [ ] GPU is available (if using GPU)
- [ ] Experiment name is set (for tracking)

During training:
- [ ] Monitor training loss (should decrease)
- [ ] Monitor validation F1 (should increase)
- [ ] Check error rate (should decrease)
- [ ] Watch for overfitting (dev loss increases while train loss decreases)

After training:
- [ ] Evaluate on dev set
- [ ] Check generated SQL quality
- [ ] Compare F1 scores
- [ ] Generate test predictions
- [ ] Save best model checkpoint

---

## Quick Start Command (Copy and Run)

```bash
python train_t5.py \
  --finetune \
  --learning_rate 3e-4 \
  --batch_size 16 \
  --max_n_epochs 10 \
  --patience_epochs 3 \
  --scheduler_type cosine \
  --num_warmup_epochs 1 \
  --weight_decay 0.01 \
  --num_beams 5 \
  --repetition_penalty 1.3 \
  --no_repeat_ngram_size 4 \
  --length_penalty 1.0 \
  --experiment_name t5_ft_improved
```

This should give you good results! Monitor the F1 score and adjust hyperparameters if needed.

