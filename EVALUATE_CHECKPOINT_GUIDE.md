# Guide: Running Checkpoints with Different Decoding Strategies

This guide shows how to evaluate saved checkpoints with different decoding strategies using `evaluate_checkpoint.py`.

## Basic Usage

```bash
python evaluate_checkpoint.py \
    --experiment_name <your_experiment_name> \
    --checkpoint_type best  # or 'latest'
```

## Required Arguments

- `--experiment_name`: Name of your experiment (matches the checkpoint folder name)
- `--checkpoint_type`: Either `best` (best_model.pt) or `latest` (checkpoint.pt)

## Decoding Strategy Examples

### 1. Beam Search (Default)

**Standard beam search:**
```bash
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --num_beams 5 \
    --length_penalty 0.8 \
    --repetition_penalty 1.2 \
    --no_repeat_ngram_size 3
```

**Beam search with strong repetition control:**
```bash
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --num_beams 5 \
    --length_penalty 1.0 \
    --repetition_penalty 1.5 \
    --no_repeat_ngram_size 3
```

**Beam search favoring shorter outputs:**
```bash
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --num_beams 5 \
    --length_penalty 1.2 \
    --repetition_penalty 1.2
```

**Beam search with constrained decoding:**
```bash
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --num_beams 5 \
    --length_penalty 0.8 \
    --repetition_penalty 1.2 \
    --use_constrained_decoding
```

### 2. Sampling Strategies

**Top-k sampling:**
```bash
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --use_sampling \
    --temperature 0.7 \
    --top_k 50 \
    --top_p 0.95
```

**Nucleus (top-p) sampling:**
```bash
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --use_sampling \
    --temperature 0.8 \
    --top_p 0.9
```

**More diverse sampling:**
```bash
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --use_sampling \
    --temperature 1.0 \
    --top_k 100 \
    --top_p 0.95
```

**More conservative sampling:**
```bash
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --use_sampling \
    --temperature 0.5 \
    --top_k 20 \
    --top_p 0.8
```

### 3. Greedy Decoding (num_beams=1)

```bash
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --num_beams 1 \
    --repetition_penalty 1.2
```

### 4. Advanced Combinations

**Beam search + constrained decoding + repetition control:**
```bash
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --num_beams 5 \
    --length_penalty 0.8 \
    --repetition_penalty 1.3 \
    --no_repeat_ngram_size 3 \
    --max_length 1024 \
    --use_constrained_decoding
```

**Sampling + constrained decoding:**
```bash
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --use_sampling \
    --temperature 0.7 \
    --top_k 50 \
    --top_p 0.95 \
    --use_constrained_decoding
```

## Parameter Descriptions

### Beam Search Parameters
- `--num_beams`: Number of beams (1 = greedy, 3-10 = typical beam search)
- `--length_penalty`: Penalty for length (0.8 = favor shorter, 1.0 = neutral, >1.0 = favor longer)
- `--repetition_penalty`: Penalty for repeating tokens (1.0 = no penalty, >1.0 = reduce repetition)
- `--no_repeat_ngram_size`: Block repeating n-grams (0 = no blocking, 3 = block 3-grams)

### Sampling Parameters
- `--use_sampling`: Enable sampling instead of beam search
- `--temperature`: Randomness (0.1 = very deterministic, 1.0 = more diverse)
- `--top_k`: Sample only from top-k tokens (50 = typical)
- `--top_p`: Nucleus sampling - sample from tokens with cumulative probability ≤ top_p (0.95 = typical)

### Other Parameters
- `--max_length`: Maximum generated sequence length (default: 768)
- `--use_constrained_decoding`: Enable SQL grammar constraints (blocks invalid tokens)
- `--batch_size`: Batch size for evaluation (default: 16)

## Comparing Strategies

To compare different strategies, run multiple evaluations with different parameters:

```bash
# Strategy 1: Beam search (5 beams)
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --num_beams 5 \
    --length_penalty 0.8 \
    --repetition_penalty 1.2

# Strategy 2: Beam search with constrained decoding
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --num_beams 5 \
    --length_penalty 0.8 \
    --repetition_penalty 1.2 \
    --use_constrained_decoding

# Strategy 3: Sampling
python evaluate_checkpoint.py \
    --experiment_name t5_ft_revised \
    --checkpoint_type best \
    --use_sampling \
    --temperature 0.7 \
    --top_k 50
```

Each run will:
1. Load the checkpoint
2. Generate SQL queries on the dev set
3. Compute metrics (Record F1, Record EM, SQL EM, Error Rate)
4. Save results to:
   - `results/t5_ft_<experiment_name>_dev_eval.sql`
   - `records/t5_ft_<experiment_name>_dev_eval.pkl`

## Finding Your Experiment Name

Your experiment name is the folder name inside `checkpoints/ft_experiments/` or `checkpoints/scr_experiments/`:

```bash
ls checkpoints/ft_experiments/
# Example output:
# t5_ft_revised
# t5_ft_baseline
# my_experiment_name
```

Then use that name as `--experiment_name`.

## Tips

1. **Start with beam search**: Beam search (num_beams=5) usually works best for SQL generation
2. **Adjust repetition_penalty**: If you see repeated conditions, increase `--repetition_penalty` to 1.3-1.5
3. **Use constrained decoding**: If you see malformed SQL (wrong order of keywords), try `--use_constrained_decoding`
4. **Compare results**: Run multiple strategies and compare their F1 scores
5. **Test set**: To evaluate on test set, you'll need to modify the script or use a similar approach

