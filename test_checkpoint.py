#!/usr/bin/env python3
"""
Script to run test inference on a saved checkpoint.

Usage:
    python test_checkpoint.py --experiment_name t5_ft_revised --checkpoint_type best
    python test_checkpoint.py --experiment_name t5_ft_revised --checkpoint_type latest --num_beams 5
"""

import os
import argparse
import torch

from t5_utils import initialize_model, load_model_from_checkpoint
from load_data import load_t5_data
from train_t5 import test_inference, get_generation_kwargs, get_args, DEVICE
from constrained_decoding import create_sql_constrained_processor

def run_test_inference(experiment_name, checkpoint_type='best', finetune=True, 
                       batch_size=16, **generation_kwargs):
    """
    Run test inference on a saved checkpoint.
    
    Args:
        experiment_name: Name of the experiment (used to find checkpoint directory)
        checkpoint_type: 'best' or 'latest' (default: 'best')
        finetune: Whether model was fine-tuned (default: True)
        batch_size: Batch size for inference (default: 16)
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
    args.max_length = generation_kwargs.get('max_length', 768)
    args.num_beams = generation_kwargs.get('num_beams', 4)
    args.repetition_penalty = generation_kwargs.get('repetition_penalty', 1.0)
    args.no_repeat_ngram_size = generation_kwargs.get('no_repeat_ngram_size', 0)
    args.length_penalty = generation_kwargs.get('length_penalty', 0.8)
    args.use_sampling = generation_kwargs.get('use_sampling', False)
    args.temperature = generation_kwargs.get('temperature', 0.7)
    args.top_k = generation_kwargs.get('top_k', 50)
    args.top_p = generation_kwargs.get('top_p', 0.95)
    args.use_constrained_decoding = generation_kwargs.get('use_constrained_decoding', False)
    
    # Load test data
    print("Loading test data...")
    _, _, test_loader = load_t5_data(batch_size, batch_size)
    
    # Load model from checkpoint
    print(f"\nLoading {checkpoint_type} checkpoint from experiment: {experiment_name}")
    
    # First, check if checkpoint files exist and their sizes
    model_type = 'ft' if finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', experiment_name)
    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    latest_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    
    # Check which checkpoint to use
    use_best = (checkpoint_type == 'best')
    if use_best and os.path.exists(best_path):
        file_size = os.path.getsize(best_path)
        if file_size < 1000:  # Likely corrupted
            print(f"⚠️  best_model.pt is suspiciously small ({file_size} bytes). Trying latest checkpoint...")
            use_best = False
    
    try:
        model = load_model_from_checkpoint(args, best=use_best)
    except Exception as e:
        print(f"\n⚠️  Failed to load checkpoint: {e}")
        if use_best and os.path.exists(latest_path):
            print("Trying to load 'latest' checkpoint instead...")
            try:
                model = load_model_from_checkpoint(args, best=False)
                print("✓ Successfully loaded latest checkpoint")
            except Exception as e2:
                print(f"⚠️  Latest checkpoint also failed: {e2}")
                raise RuntimeError("Both checkpoints are corrupted. Please retrain the model.")
        else:
            raise RuntimeError(f"Checkpoint loading failed: {e}")
    
    model.eval()
    
    # Set up paths - include decoding strategy in filename to avoid overwrites
    model_type = 'ft' if finetune else 'scr'
    
    # Create unique filename suffix based on decoding strategy
    strategy_suffix = []
    if generation_kwargs.get('use_constrained_decoding'):
        strategy_suffix.append('constrained')
    
    if generation_kwargs.get('use_sampling') or generation_kwargs.get('num_beams', 0) == 0:
        strategy_suffix.append(f"sampling_t{generation_kwargs.get('temperature', 0.7)}")
        if generation_kwargs.get('top_k'):
            strategy_suffix.append(f"k{generation_kwargs.get('top_k')}")
        if generation_kwargs.get('top_p'):
            strategy_suffix.append(f"p{generation_kwargs.get('top_p')}")
    else:
        num_beams = generation_kwargs.get('num_beams', 4)
        if num_beams == 1:
            strategy_suffix.append('greedy')
        else:
            strategy_suffix.append(f"beam{num_beams}")
        if generation_kwargs.get('length_penalty') != 0.8:  # Only include if non-default
            strategy_suffix.append(f"lp{generation_kwargs.get('length_penalty', 0.8)}")
    
    if generation_kwargs.get('repetition_penalty', 1.0) != 1.0:
        strategy_suffix.append(f"rep{generation_kwargs.get('repetition_penalty', 1.0)}")
    
    if strategy_suffix:
        strategy_str = '_' + '_'.join(str(strategy_suffix))
    else:
        strategy_str = ''
    
    model_sql_path = os.path.join('results', f't5_{model_type}_{experiment_name}{strategy_str}_test.sql')
    model_record_path = os.path.join('records', f't5_{model_type}_{experiment_name}{strategy_str}_test.pkl')
    
    # Run test inference
    print(f"\nRunning test inference...")
    print(f"Generation settings:")
    print(f"  - max_length: {args.max_length}")
    print(f"  - num_beams: {args.num_beams}")
    print(f"  - repetition_penalty: {args.repetition_penalty}")
    print(f"  - no_repeat_ngram_size: {args.no_repeat_ngram_size}")
    print(f"  - length_penalty: {args.length_penalty}")
    if args.use_constrained_decoding:
        print(f"  - constrained_decoding: enabled")
    print()
    
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    
    # Print results
    print("\n" + "="*80)
    print("TEST INFERENCE COMPLETE")
    print("="*80)
    print(f"Checkpoint: {checkpoint_type} ({experiment_name})")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  SQL: {model_sql_path}")
    print(f"  Records: {model_record_path}")
    
    return {
        'sql_path': model_sql_path,
        'record_path': model_record_path
    }

def main():
    parser = argparse.ArgumentParser(description='Run test inference on a saved checkpoint')
    
    # Required arguments
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment (used to find checkpoint)')
    parser.add_argument('--checkpoint_type', type=str, default='best',
                        choices=['best', 'latest'],
                        help='Which checkpoint to load: "best" (best_model.pt) or "latest" (checkpoint.pt)')
    
    # Model arguments
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Whether model was fine-tuned (default: False, set --finetune for fine-tuned models)')
    parser.add_argument('--from_scratch', action='store_true', default=False,
                        help='Whether model was trained from scratch (default: False, set --from_scratch for scratch models)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    
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
    parser.add_argument('--max_length', type=int, default=768,
                        help='Maximum length of generated sequences (default: 768)')
    parser.add_argument('--use_constrained_decoding', action='store_true',
                        help='Use constrained decoding to ensure valid SQL generation')
    
    args = parser.parse_args()
    
    # Handle finetune/from_scratch logic
    if args.from_scratch:
        args.finetune = False
    # If neither is set, default to finetune=True (most common case)
    if not args.finetune and not args.from_scratch:
        args.finetune = True
        print("ℹ️  Defaulting to --finetune=True (use --from_scratch if your model was trained from scratch)")
    
    # Extract generation kwargs
    generation_kwargs = {
        'max_length': args.max_length,
        'num_beams': args.num_beams,
        'repetition_penalty': args.repetition_penalty,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
        'length_penalty': args.length_penalty,
        'use_sampling': args.use_sampling,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'use_constrained_decoding': args.use_constrained_decoding,
    }
    
    # Run test inference
    results = run_test_inference(
        experiment_name=args.experiment_name,
        checkpoint_type=args.checkpoint_type,
        finetune=args.finetune,
        batch_size=args.batch_size,
        **generation_kwargs
    )
    
    return results

if __name__ == "__main__":
    main()

