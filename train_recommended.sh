#!/bin/bash
# Recommended training script for T5 SQL generation
# This script uses optimal hyperparameters for fine-tuning T5

echo "Starting T5 training with recommended parameters..."
echo "=================================================="

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

echo ""
echo "Training complete!"
echo "Check results in:"
echo "  - Dev SQL: results/t5_ft_t5_ft_revised_dev.sql"
echo "  - Test SQL: results/t5_ft_t5_ft_revised_test.sql"
echo "  - Training curves: checkpoints/ft_experiments/t5_ft_revised/training_curves.png"

