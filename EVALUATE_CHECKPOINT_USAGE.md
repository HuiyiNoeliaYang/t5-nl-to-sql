# How to Use evaluate_checkpoint.py

## Basic Usage

### 1. Evaluate Best Model (Most Common)

```bash
python evaluate_checkpoint.py --experiment_name t5_ft_revised
```

This will:
- Load `best_model.pt` from `checkpoints/ft_experiments/t5_ft_revised/`
- Evaluate on dev set
- Save results to `results/t5_ft_t5_ft_revised_dev_eval.sql`

### 2. Evaluate Latest Checkpoint

```bash
python evaluate_checkpoint.py --experiment_name t5_ft_revised --checkpoint_type latest
```

This will:
- Load `checkpoint.pt` (latest epoch) instead of `best_model.pt`

### 3. With Custom Generation Parameters

```bash
python evaluate_checkpoint.py \
  --experiment_name t5_ft_revised \
  --num_beams 5 \
  --repetition_penalty 1.2 \
  --no_repeat_ngram_size 3 \
  --length_penalty 0.8
```

## Common Examples

### Example 1: Evaluate Best Model with Default Settings

```bash
python evaluate_checkpoint.py --experiment_name t5_ft_revised
```

**Output:**
```
Loading dev data...
Loading best checkpoint from experiment: t5_ft_revised
Evaluating on dev set...
Generation settings:
  - num_beams: 4
  - repetition_penalty: 1.0
  - no_repeat_ngram_size: 0
  - length_penalty: 0.8

================================================================================
EVALUATION RESULTS
================================================================================
Checkpoint: best (t5_ft_revised)
Dev Loss: 0.0700
Record F1: 0.5200
Record EM: 0.3000
SQL EM: 0.1500
Error Rate: 12.00%
================================================================================

Results saved to:
  SQL: results/t5_ft_t5_ft_revised_dev_eval.sql
  Records: records/t5_ft_t5_ft_revised_dev_eval.pkl
```

### Example 2: Compare Different Generation Settings

**Test with greedy decoding:**
```bash
python evaluate_checkpoint.py \
  --experiment_name t5_ft_revised \
  --num_beams 1 \
  --repetition_penalty 1.0 \
  --no_repeat_ngram_size 0
```

**Test with beam search:**
```bash
python evaluate_checkpoint.py \
  --experiment_name t5_ft_revised \
  --num_beams 5 \
  --repetition_penalty 1.2 \
  --no_repeat_ngram_size 3
```

**Test with sampling:**
```bash
python evaluate_checkpoint.py \
  --experiment_name t5_ft_revised \
  --use_sampling \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 0.95
```

### Example 3: Evaluate Model Trained from Scratch

```bash
python evaluate_checkpoint.py \
  --experiment_name t5_scratch_experiment \
  --finetune False
```

**Note:** Use `--finetune False` if the model was trained from scratch (not fine-tuned).

## All Available Options

```bash
python evaluate_checkpoint.py \
  --experiment_name EXPERIMENT_NAME \    # Required: name of your experiment
  --checkpoint_type best \               # 'best' or 'latest' (default: 'best')
  --finetune \                           # Whether model was fine-tuned (default: True)
  --batch_size 16 \                      # Batch size (default: 16)
  --num_beams 4 \                        # Beam search size (default: 4)
  --repetition_penalty 1.0 \             # Repetition penalty (default: 1.0)
  --no_repeat_ngram_size 0 \             # N-gram blocking (default: 0)
  --length_penalty 0.8 \                 # Length penalty (default: 0.8)
  --use_sampling \                       # Use sampling instead of beam search
  --temperature 0.7 \                    # Temperature for sampling (default: 0.7)
  --top_k 50 \                           # Top-k for sampling (default: 50)
  --top_p 0.95                           # Top-p (nucleus) for sampling (default: 0.95)
```

## Finding Your Experiment Name

Your experiment name is what you set with `--experiment_name` during training. For example:

**During training:**
```bash
python train_t5.py --experiment_name t5_ft_revised ...
```

**During evaluation:**
```bash
python evaluate_checkpoint.py --experiment_name t5_ft_revised
```

## Checkpoint Locations

The script looks for checkpoints in:
- **Best model:** `checkpoints/ft_experiments/EXPERIMENT_NAME/best_model.pt`
- **Latest checkpoint:** `checkpoints/ft_experiments/EXPERIMENT_NAME/checkpoint.pt`

If you trained from scratch (not fine-tuned):
- **Best model:** `checkpoints/scr_experiments/EXPERIMENT_NAME/best_model.pt`
- **Latest checkpoint:** `checkpoints/scr_experiments/EXPERIMENT_NAME/checkpoint.pt`

## Output Files

After evaluation, results are saved to:
- **SQL queries:** `results/t5_ft_EXPERIMENT_NAME_dev_eval.sql`
- **Database records:** `records/t5_ft_EXPERIMENT_NAME_dev_eval.pkl`

These are separate from training outputs (which use `_dev.sql` without `_eval`).

## Quick Reference

| Task | Command |
|------|---------|
| Evaluate best model | `python evaluate_checkpoint.py --experiment_name NAME` |
| Evaluate latest checkpoint | `python evaluate_checkpoint.py --experiment_name NAME --checkpoint_type latest` |
| Test with greedy decoding | `python evaluate_checkpoint.py --experiment_name NAME --num_beams 1` |
| Test with different beam size | `python evaluate_checkpoint.py --experiment_name NAME --num_beams 10` |
| Test with repetition penalty | `python evaluate_checkpoint.py --experiment_name NAME --repetition_penalty 1.3` |
| Test with sampling | `python evaluate_checkpoint.py --experiment_name NAME --use_sampling` |
| Evaluate scratch model | `python evaluate_checkpoint.py --experiment_name NAME --finetune False` |

## Troubleshooting

### Error: "Checkpoint not found"

**Problem:** The checkpoint file doesn't exist.

**Solution:** 
1. Check that the experiment name matches what you used during training
2. Verify the checkpoint exists: `ls checkpoints/ft_experiments/YOUR_EXPERIMENT_NAME/`
3. Make sure you're using `--finetune False` if you trained from scratch

### Error: "CUDA out of memory"

**Solution:** Reduce batch size:
```bash
python evaluate_checkpoint.py --experiment_name NAME --batch_size 8
```

### Want to see what checkpoints are available?

```bash
# List all experiments
ls checkpoints/ft_experiments/

# List checkpoints for a specific experiment
ls checkpoints/ft_experiments/t5_ft_revised/
```

## Tips

1. **Always evaluate best model first** - It has the highest F1 score
2. **Compare generation settings** - Try different `num_beams`, `repetition_penalty`, etc.
3. **Save results with different names** - The script automatically adds `_eval` suffix
4. **Use for debugging** - Test if different generation parameters improve results

