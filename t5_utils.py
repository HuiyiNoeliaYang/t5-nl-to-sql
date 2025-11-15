import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    pass

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    model_name = "google-t5/t5-small"
    
    if args.finetune:
        # Fine-tune pretrained T5-small
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        # Train from scratch using T5-small config
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)
    
    # Freeze parameters based on arguments
    freeze_model_parameters(model, args)
    
    model.to(DEVICE)
    return model

def freeze_model_parameters(model, args):
    '''
    Freeze model parameters based on the provided arguments.
    Common strategies:
    - Freeze encoder: Keep encoder frozen, only train decoder
    - Freeze decoder: Keep decoder frozen, only train encoder
    - Freeze embeddings: Keep embedding layers frozen
    - Freeze early layers: Freeze first N layers, train later layers
    '''
    total_params = sum(p.numel() for p in model.parameters())
    
    # Freeze embeddings if requested
    if args.freeze_embeddings:
        if hasattr(model, 'shared'):
            model.shared.weight.requires_grad = False
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'embed_tokens'):
            model.encoder.embed_tokens.weight.requires_grad = False
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'embed_tokens'):
            model.decoder.embed_tokens.weight.requires_grad = False
    
    # Freeze entire encoder if requested
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    # Freeze entire decoder if requested
    if args.freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False
    
    # Freeze first N encoder layers
    if args.freeze_n_encoder_layers > 0 and hasattr(model.encoder, 'block'):
        num_layers = len(model.encoder.block)
        layers_to_freeze = min(args.freeze_n_encoder_layers, num_layers)
        for i in range(layers_to_freeze):
            for param in model.encoder.block[i].parameters():
                param.requires_grad = False
        print(f"Frozen first {layers_to_freeze} encoder layers (out of {num_layers})")
    
    # Freeze first N decoder layers
    if args.freeze_n_decoder_layers > 0 and hasattr(model.decoder, 'block'):
        num_layers = len(model.decoder.block)
        layers_to_freeze = min(args.freeze_n_decoder_layers, num_layers)
        for i in range(layers_to_freeze):
            for param in model.decoder.block[i].parameters():
                param.requires_grad = False
        print(f"Frozen first {layers_to_freeze} decoder layers (out of {num_layers})")
    
    # Print statistics
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params_after
    
    print(f"\nParameter Freezing Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params_after:,} ({100*trainable_params_after/total_params:.2f}%)")
    print(f"  Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    mkdir(checkpoint_dir)
    filename = "best_model.pt" if best else "checkpoint.pt"
    path = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), path)

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    model_name = "google-t5/t5-small"
    
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)
    
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    filename = "best_model.pt" if best else "checkpoint.pt"
    path = os.path.join(checkpoint_dir, filename)
    
    if os.path.exists(path):
        # Check file size to detect corrupted/incomplete checkpoints
        file_size = os.path.getsize(path)
        if file_size == 0:
            print(f"WARNING: Checkpoint file {path} is empty (0 bytes). Skipping load.")
        elif file_size < 1000:  # Very small file is likely corrupted
            print(f"WARNING: Checkpoint file {path} is suspiciously small ({file_size} bytes). It may be corrupted.")
        else:
            try:
                checkpoint = torch.load(path, map_location=DEVICE)
                model.load_state_dict(checkpoint)
                print(f"✓ Successfully loaded checkpoint from {path} ({file_size / (1024*1024):.2f} MB)")
            except Exception as e:
                print(f"ERROR: Failed to load checkpoint from {path}")
                print(f"  Error: {e}")
                print(f"  File size: {file_size / (1024*1024):.2f} MB")
                print(f"  The checkpoint file may be corrupted or incomplete.")
                print(f"  Continuing with pretrained weights only (not fine-tuned).")
    else:
        print(f"WARNING: Checkpoint file not found: {path}")
        print(f"  Continuing with pretrained weights only (not fine-tuned).")
    
    # Apply freezing again (in case we're loading for evaluation)
    freeze_model_parameters(model, args)
    
    model.to(DEVICE)
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

