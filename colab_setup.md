# Google Colab Setup Guide

## Quick Setup for Google Colab

### Option 1: Install packages one by one (Recommended for Colab)

```python
# Run this in a Colab cell
!pip install numpy torch tokenizers transformers accelerate nltk tqdm sentencepiece
!pip install wandb matplotlib plotly seaborn  # Optional: for visualization
```

### Option 2: Install with flexible versions

```python
# Run this in a Colab cell
!pip install numpy>=1.24.0 torch>=2.0.0 tokenizers>=0.13.0 transformers>=4.30.0
!pip install accelerate>=0.20.0 nltk>=3.8.0 tqdm>=4.65.0 sentencepiece>=0.1.0
```

### Option 3: Install from requirements.txt (if it fails, use Option 1)

```python
# First, upload requirements.txt to Colab, then:
!pip install -r requirements.txt
```

### Option 4: Install specific compatible versions for Colab

```python
!pip install numpy==1.24.3 torch==2.0.1 tokenizers==0.13.3 transformers==4.30.2
!pip install accelerate==0.20.3 nltk==3.8.1 tqdm==4.65.0 sentencepiece==0.1.99
```

## Common Issues and Solutions

### Issue 1: Package version conflicts
**Solution**: Install without version pins first, then specific packages if needed
```python
!pip install --upgrade pip
!pip install numpy torch transformers tokenizers accelerate nltk tqdm sentencepiece
```

### Issue 2: bitsandbytes not available
**Solution**: Skip it (it's only for quantization, not needed for basic training)
```python
# Just don't install bitsandbytes - it's optional
```

### Issue 3: CUDA/GPU issues
**Solution**: Colab usually handles this automatically, but you can check:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Recommended Colab Setup Cell

```python
# Run this cell first in Colab
import sys
print(f"Python version: {sys.version}")

# Install core packages
!pip install -q numpy torch tokenizers transformers accelerate nltk tqdm sentencepiece

# Verify installation
try:
    import torch
    import transformers
    from transformers import T5TokenizerFast, T5ForConditionalGeneration
    print("✓ All packages installed successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Import error: {e}")
```

