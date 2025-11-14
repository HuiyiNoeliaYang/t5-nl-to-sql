# Model Saving Based on F1 Score

## Overview

Your training code **already saves models based on the best F1 score**, not loss. This is the correct approach for your task since F1 is the primary evaluation metric.

## How It Works

### 1. Tracking Best F1

```python
best_f1 = -1  # Initialize to -1 (line 81)
epochs_since_improvement = 0
```

### 2. Comparing F1 After Each Epoch

After each epoch, the code:
1. Evaluates on dev set and gets `record_f1`
2. Compares to `best_f1`:

```python
if record_f1 > best_f1:
    best_f1 = record_f1
    epochs_since_improvement = 0
    # Save best model
else:
    epochs_since_improvement += 1
    # Don't save (F1 didn't improve)
```

### 3. Saving Models

**Every epoch:**
- Saves a checkpoint: `checkpoint.pt` (for resuming training)

**Only when F1 improves:**
- Saves best model: `best_model.pt` (the one with highest F1)

```python
save_model(checkpoint_dir, model, best=False)  # Every epoch
if epochs_since_improvement == 0:  # F1 just improved
    save_model(checkpoint_dir, model, best=True)  # Save best
```

### 4. Early Stopping

Training stops when F1 doesn't improve for `patience_epochs` consecutive epochs:

```python
if epochs_since_improvement >= args.patience_epochs:
    break  # Stop training
```

### 5. Loading Best Model

After training, the code automatically loads the best model:

```python
model = load_model_from_checkpoint(args, best=True)  # Loads best_model.pt
```

## Example Training Output

With the improved logging, you'll now see:

```
Epoch 0: Dev loss: 0.5000, Record F1: 0.1500, Record EM: 0.0500, SQL EM: 0.0200
Epoch 0: New best F1! (0.1500) - Saving best model...

Epoch 1: Dev loss: 0.2000, Record F1: 0.2500, Record EM: 0.1000, SQL EM: 0.0500
Epoch 1: New best F1! (0.2500) - Saving best model...

Epoch 2: Dev loss: 0.1200, Record F1: 0.3500, Record EM: 0.1500, SQL EM: 0.0800
Epoch 2: New best F1! (0.3500) - Saving best model...

Epoch 3: Dev loss: 0.0900, Record F1: 0.4200, Record EM: 0.2000, SQL EM: 0.1000
Epoch 3: New best F1! (0.4200) - Saving best model...

Epoch 4: Dev loss: 0.0800, Record F1: 0.4800, Record EM: 0.2500, SQL EM: 0.1200
Epoch 4: New best F1! (0.4800) - Saving best model...

Epoch 5: Dev loss: 0.0700, Record F1: 0.5200, Record EM: 0.3000, SQL EM: 0.1500
Epoch 5: New best F1! (0.5200) - Saving best model...

Epoch 6: Dev loss: 0.0700, Record F1: 0.5100, Record EM: 0.2900, SQL EM: 0.1400
Epoch 6: F1 did not improve (best: 0.5200, current: 0.5100, patience: 1/5)

Epoch 7: Dev loss: 0.0700, Record F1: 0.5000, Record EM: 0.2800, SQL EM: 0.1300
Epoch 7: F1 did not improve (best: 0.5200, current: 0.5000, patience: 2/5)

...

Epoch 10: Dev loss: 0.0700, Record F1: 0.4900, Record EM: 0.2700, SQL EM: 0.1200
Epoch 10: F1 did not improve (best: 0.5200, current: 0.4900, patience: 5/5)

Early stopping: No F1 improvement for 5 epochs.
Best F1 score: 0.5200

================================================================================
Loading best model (based on highest validation F1 score)...
================================================================================
```

## Why F1 Instead of Loss?

### Loss Can Be Misleading

- **Low loss ≠ Good F1**: Model can have low loss but poor F1
- **Loss measures token-level accuracy**: Doesn't capture SQL correctness
- **F1 measures SQL correctness**: What you actually care about

### F1 is the Right Metric

- **F1 measures record-level accuracy**: Does the SQL return correct results?
- **Better for structured outputs**: SQL needs to be semantically correct
- **Matches your evaluation**: Final evaluation uses F1

## File Locations

After training, you'll find:

```
checkpoints/
  ft_experiments/
    your_experiment_name/
      checkpoint.pt      # Latest epoch (for resuming)
      best_model.pt      # Best F1 model (for evaluation) ← Use this!
```

## Verification

To verify the best model is being used:

1. **Check training logs**: Look for "New best F1!" messages
2. **Check saved files**: `best_model.pt` should exist
3. **Check final evaluation**: Should match the best F1 from training

## Summary

✅ **Model saving is already based on F1** (not loss)  
✅ **Best model is automatically saved** when F1 improves  
✅ **Best model is automatically loaded** for final evaluation  
✅ **Early stopping uses F1** (not loss)  
✅ **Clear logging** shows when best model is saved  

You don't need to change anything - it's already working correctly! The improved logging just makes it clearer what's happening.

