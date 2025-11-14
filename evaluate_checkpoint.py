#!/usr/bin/env python3
"""
Script to evaluate a saved checkpoint on the dev set.

Usage:
    python evaluate_checkpoint.py --experiment_name t5_ft_revised --checkpoint_type best
    python evaluate_checkpoint.py --experiment_name t5_ft_revised --checkpoint_type latest
"""

import os
import argparse
import torch

from t5_utils import initialize_model, load_model_from_checkpoint
from load_data import load_t5_data
from train_t5 import eval_epoch, get_generation_kwargs, get_args, DEVICE

def evaluate_checkpoint(experiment_name, checkpoint_type='best', finetune=True, 
                       batch_size=16, **generation_kwargs):
    """
    Evaluate a saved checkpoint on the dev set.
    
    Args:
        experiment_name: Name of the experiment (used to find checkpoint directory)
        checkpoint_type: 'best' or 'latest' (default: 'best')
        finetune: Whether model was fine-tuned (default: True)
        batch_size: Batch size for evaluation (default: 16)
        **generation_kwargs: Generation parameters (num_beams, repetition_penalty, etc.)
    """
    
    # Create minimal args object for loading model
    args = argparse.Namespace()
    args.finetune = finetune
    args.freeze_encoder = False
    args.freeze_decoder = False
    args.freeze_embeddings = False
    args.freeze_n_encoder_layers = 0
    args.freeze_n_decoder_layers = 0
    args.experiment_name = experiment_name
    
    # Set generation parameters (use defaults if not provided)
    args.num_beams = generation_kwargs.get('num_beams', 4)
    args.repetition_penalty = generation_kwargs.get('repetition_penalty', 1.0)
    args.no_repeat_ngram_size = generation_kwargs.get('no_repeat_ngram_size', 0)
    args.length_penalty = generation_kwargs.get('length_penalty', 0.8)
    args.use_sampling = generation_kwargs.get('use_sampling', False)
    args.temperature = generation_kwargs.get('temperature', 0.7)
    args.top_k = generation_kwargs.get('top_k', 50)
    args.top_p = generation_kwargs.get('top_p', 0.95)
    
    # Load data
    print("Loading dev data...")
    _, dev_loader, _ = load_t5_data(batch_size, batch_size)
    
    # Load model from checkpoint
    print(f"\nLoading {checkpoint_type} checkpoint from experiment: {experiment_name}")
    model = load_model_from_checkpoint(args, best=(checkpoint_type == 'best'))
    model.eval()
    
    # Set up paths
    model_type = 'ft' if finetune else 'scr'
    gt_sql_path = os.path.join('data', 'dev.sql')
    gt_record_path = os.path.join('records', 'ground_truth_dev.pkl')
    model_sql_path = os.path.join('results', f't5_{model_type}_{experiment_name}_dev_eval.sql')
    model_record_path = os.path.join('records', f't5_{model_type}_{experiment_name}_dev_eval.pkl')
    
    # Evaluate
    print(f"\nEvaluating on dev set...")
    print(f"Generation settings:")
    print(f"  - num_beams: {args.num_beams}")
    print(f"  - repetition_penalty: {args.repetition_penalty}")
    print(f"  - no_repeat_ngram_size: {args.no_repeat_ngram_size}")
    print(f"  - length_penalty: {args.length_penalty}")
    print()
    
    dev_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
        args, model, dev_loader,
        gt_sql_path, model_sql_path,
        gt_record_path, model_record_path
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Checkpoint: {checkpoint_type} ({experiment_name})")
    print(f"Dev Loss: {dev_loss:.4f}")
    print(f"Record F1: {record_f1:.4f}")
    print(f"Record EM: {record_em:.4f}")
    print(f"SQL EM: {sql_em:.4f}")
    print(f"Error Rate: {error_rate*100:.2f}%")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  SQL: {model_sql_path}")
    print(f"  Records: {model_record_path}")
    
    return {
        'loss': dev_loss,
        'record_f1': record_f1,
        'record_em': record_em,
        'sql_em': sql_em,
        'error_rate': error_rate
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate a saved checkpoint on dev set')
    
    # Required arguments
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment (used to find checkpoint)')
    parser.add_argument('--checkpoint_type', type=str, default='best',
                        choices=['best', 'latest'],
                        help='Which checkpoint to load: "best" (best_model.pt) or "latest" (checkpoint.pt)')
    
    # Model arguments
    parser.add_argument('--finetune', action='store_true', default=True,
                        help='Whether model was fine-tuned (default: True)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    
    # Generation arguments
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Number of beams for beam search')
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help='Repetition penalty (1.0 = no penalty)')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0,
                        help='Prevent repeating n-grams (0 = no blocking)')
    parser.add_argument('--length_penalty', type=float, default=0.8,
                        help='Length penalty for beam search')
    parser.add_argument('--use_sampling', action='store_true',
                        help='Use sampling instead of beam search')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k for sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p (nucleus) for sampling')
    
    args = parser.parse_args()
    
    # Extract generation kwargs
    generation_kwargs = {
        'num_beams': args.num_beams,
        'repetition_penalty': args.repetition_penalty,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
        'length_penalty': args.length_penalty,
        'use_sampling': args.use_sampling,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
    }
    
    # Evaluate
    results = evaluate_checkpoint(
        experiment_name=args.experiment_name,
        checkpoint_type=args.checkpoint_type,
        finetune=args.finetune,
        batch_size=args.batch_size,
        **generation_kwargs
    )
    
    return results

if __name__ == "__main__":
    main()

