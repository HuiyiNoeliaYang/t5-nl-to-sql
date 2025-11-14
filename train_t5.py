import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb
import matplotlib.pyplot as plt

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    parser.add_argument('--freeze_encoder', action='store_true', help="Freeze encoder parameters")
    parser.add_argument('--freeze_decoder', action='store_true', help="Freeze decoder parameters")
    parser.add_argument('--freeze_embeddings', action='store_true', help="Freeze embedding layers")
    parser.add_argument('--freeze_n_encoder_layers', type=int, default=0,
                        help="Freeze first N encoder layers (0 = no freezing)")
    parser.add_argument('--freeze_n_decoder_layers', type=int, default=0,
                        help="Freeze first N decoder layers (0 = no freezing)")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    
    # Generation hyperparameters
    parser.add_argument('--num_beams', type=int, default=3,
                        help="Number of beams for beam search (1 = greedy decoding, 0 = use sampling)")
    parser.add_argument('--repetition_penalty', type=float, default=1.2,
                        help="Penalty for repeating tokens (1.0 = no penalty)")
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3,
                        help="Prevent repeating n-grams of this size")
    parser.add_argument('--length_penalty', type=float, default=1.0,
                        help="Penalty for longer sequences (1.0 = no penalty)")
    parser.add_argument('--use_sampling', action='store_true',
                        help="Use sampling instead of beam search (overrides num_beams)")
    parser.add_argument('--temperature', type=float, default=0.7,
                        help="Temperature for sampling (lower = more deterministic, higher = more diverse)")
    parser.add_argument('--top_k', type=int, default=50,
                        help="Top-k sampling: only sample from top k tokens")
    parser.add_argument('--top_p', type=float, default=0.95,
                        help="Nucleus sampling: only sample from tokens with cumulative probability <= top_p")
    parser.add_argument('--max_length', type=int, default=768,
                        help="Maximum length of generated sequences (default: 768)")

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0
    
    # Store losses for plotting
    train_losses = []
    dev_losses = []
    epoch_numbers = []

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        train_losses.append(tr_loss)
        epoch_numbers.append(epoch)
        print(f"Epoch {epoch}: Average train loss was {tr_loss:.4f}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        dev_losses.append(eval_loss)
        print(f"Epoch {epoch}: Dev loss: {eval_loss:.4f}, Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
        
        # Plot loss after each epoch
        plot_loss_curves(epoch_numbers, train_losses, dev_losses, checkpoint_dir)

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def plot_loss_curves(epochs, train_losses, dev_losses, save_dir):
    '''
    Plot training and validation loss curves after each epoch.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    plt.plot(epochs, dev_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot (overwrites after each epoch)
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Display plot in Colab/Jupyter
    try:
        # Try to detect Colab
        try:
            import google.colab
            # In Colab: display using IPython
            from IPython.display import Image, display
            # Close the figure first to avoid the "Figure(1000x600)" output
            plt.close()
            # Then display the saved image
            display(Image(plot_path))
            print(f"\n📊 Plot updated! File: {plot_path}")
        except ImportError:
            # Not in Colab
            import sys
            if 'ipykernel' in sys.modules:
                # In Jupyter
                plt.show()
                plt.close()
            else:
                # For terminal: try non-blocking display
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()
    except Exception as e:
        # If display fails, just save the file and print path
        plt.close()
        print(f"\n📊 Plot saved to: {plot_path}")
        print(f"   View it in Colab's file browser at: {plot_path}")

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens

def get_generation_kwargs(args, tokenizer):
    '''
    Helper function to get generation parameters.
    Centralizes generation config to avoid duplication.
    Supports both beam search and sampling strategies.
    
    Note: These parameters are NOT implemented in this code - they're passed to
    HuggingFace Transformers' model.generate() method, which implements them internally.
    See GENERATION_PARAMETERS_EXPLAINED.md for details on how they work.
    
    Parameters:
    - repetition_penalty: Applied at each step to reduce probability of tokens 
      already in the generated sequence (1.0 = no penalty, >1.0 = penalty)
    - no_repeat_ngram_size: Completely blocks n-grams from repeating 
      (0 = no blocking, 3 = block repeating 3-grams)
    - length_penalty: Only for beam search - normalizes beam scores by sequence length
      (1.0 = divide by length, <1.0 = favor longer, >1.0 = favor shorter)
    '''
    base_kwargs = {
        'max_length': args.max_length,
        'repetition_penalty': args.repetition_penalty,  # Reduces probability of repeated tokens
        'no_repeat_ngram_size': args.no_repeat_ngram_size,  # Blocks duplicate n-grams
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    
    if args.use_sampling or args.num_beams == 0:
        # Use sampling strategy
        base_kwargs.update({
            'do_sample': True,
            'temperature': args.temperature,
            'top_k': args.top_k,
            'top_p': args.top_p,
        })
        print(f"Using sampling: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    else:
        # Use beam search strategy
        base_kwargs.update({
            'num_beams': args.num_beams,
            'early_stopping': True,
            'length_penalty': args.length_penalty,
            'do_sample': False,
        })
        print(f"Using beam search: num_beams={args.num_beams}, length_penalty={args.length_penalty}")
    
    return base_kwargs
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    generated_queries = []
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            
            # Compute loss
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            
            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Generate SQL queries
            generation_kwargs = get_generation_kwargs(args, tokenizer)
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                **generation_kwargs
            )
            
            # Decode generated queries
            for gen in generated:
                sql = tokenizer.decode(gen, skip_special_tokens=True)
                generated_queries.append(sql)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    # Save generated queries and compute metrics
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    
    error_rate = sum(1 for msg in error_msgs if msg) / len(error_msgs) if error_msgs else 0
    
    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    generated_queries = []
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate SQL queries
            generation_kwargs = get_generation_kwargs(args, tokenizer)
            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                **generation_kwargs
            )
            
            # Decode generated queries
            for gen in generated:
                sql = tokenizer.decode(gen, skip_special_tokens=True)
                generated_queries.append(sql)
    
    # Save generated queries and compute records
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    dev_loss, dev_record_em, dev_record_f1, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_test.pkl')
    print(f"\nGenerating test set predictions...")
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    print(f"\n✓ Training complete! Test predictions saved to:")
    print(f"  SQL: {model_sql_path}")
    print(f"  Records: {model_record_path}")

if __name__ == "__main__":
    main()
