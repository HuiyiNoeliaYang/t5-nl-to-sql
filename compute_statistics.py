"""
Script to compute data statistics using T5 tokenizer for Q4.
Reports statistics before and after preprocessing.
"""
import os
import numpy as np
from transformers import T5TokenizerFast
from load_data import load_lines

def compute_statistics(texts, tokenizer, split_name="", preprocessed=False):
    """
    Compute statistics for a list of texts using T5 tokenizer.
    
    Args:
        texts: List of text strings
        tokenizer: T5TokenizerFast instance
        split_name: Name of the split (e.g., "train", "dev")
        preprocessed: Whether this is preprocessed data
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        "split": split_name,
        "preprocessed": preprocessed,
        "num_samples": len(texts),
    }
    
    # Tokenize all texts
    tokenized = tokenizer(texts, return_length=True, padding=False, truncation=False)
    
    # Get sequence lengths
    lengths = tokenized['length']
    stats["avg_length"] = np.mean(lengths)
    stats["median_length"] = np.median(lengths)
    stats["min_length"] = int(np.min(lengths))
    stats["max_length"] = int(np.max(lengths))
    stats["std_length"] = np.std(lengths)
    
    # Percentiles
    stats["p25_length"] = np.percentile(lengths, 25)
    stats["p75_length"] = np.percentile(lengths, 75)
    stats["p95_length"] = np.percentile(lengths, 95)
    
    # Count sequences that would be truncated at different max_lengths
    stats["truncated_at_128"] = np.sum(lengths > 128)
    stats["truncated_at_256"] = np.sum(lengths > 256)
    stats["truncated_at_512"] = np.sum(lengths > 512)
    
    # Vocabulary statistics
    all_token_ids = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        all_token_ids.extend(tokens)
    
    unique_tokens = len(set(all_token_ids))
    stats["unique_tokens"] = unique_tokens
    stats["vocab_size"] = len(tokenizer.get_vocab())
    
    # Token frequency (top tokens)
    from collections import Counter
    token_counts = Counter(all_token_ids)
    stats["most_common_tokens"] = token_counts.most_common(10)
    
    return stats

def print_statistics(stats, tokenizer):
    """Print statistics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Statistics for {stats['split']} set ({'Preprocessed' if stats['preprocessed'] else 'Raw'})")
    print(f"{'='*60}")
    print(f"Number of samples: {stats['num_samples']}")
    print(f"\nSequence Length Statistics (in tokens):")
    print(f"  Average: {stats['avg_length']:.2f}")
    print(f"  Median: {stats['median_length']:.2f}")
    print(f"  Min: {stats['min_length']}")
    print(f"  Max: {stats['max_length']}")
    print(f"  Std Dev: {stats['std_length']:.2f}")
    print(f"  25th percentile: {stats['p25_length']:.2f}")
    print(f"  75th percentile: {stats['p75_length']:.2f}")
    print(f"  95th percentile: {stats['p95_length']:.2f}")
    print(f"\nTruncation Statistics:")
    print(f"  Sequences > 128 tokens: {stats['truncated_at_128']} ({100*stats['truncated_at_128']/stats['num_samples']:.2f}%)")
    print(f"  Sequences > 256 tokens: {stats['truncated_at_256']} ({100*stats['truncated_at_256']/stats['num_samples']:.2f}%)")
    print(f"  Sequences > 512 tokens: {stats['truncated_at_512']} ({100*stats['truncated_at_512']/stats['num_samples']:.2f}%)")
    print(f"\nVocabulary Statistics:")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    print(f"  Unique tokens in dataset: {stats['unique_tokens']}")
    print(f"\nTop 10 Most Common Token IDs:")
    for token_id, count in stats['most_common_tokens']:
        token = tokenizer.decode([token_id])
        print(f"  Token ID {token_id} ('{token}'): {count} occurrences")

def compute_vocab_size(texts, tokenizer):
    """Compute vocabulary size (unique tokens) for a set of texts."""
    all_token_ids = set()
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        all_token_ids.update(tokens)
    return len(all_token_ids)

def main():
    # Initialize T5 tokenizer
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    
    data_folder = 'data'
    splits = ['train', 'dev']
    
    # Collect statistics
    results = {}
    
    for split in splits:
        # Load raw data
        nl_path = os.path.join(data_folder, f"{split}.nl")
        sql_path = os.path.join(data_folder, f"{split}.sql")
        
        nl_texts = load_lines(nl_path)
        sql_texts = load_lines(sql_path)
        
        # Tokenize to get lengths
        nl_tokenized = tokenizer(nl_texts, return_length=True, padding=False, truncation=False)
        sql_tokenized = tokenizer(sql_texts, return_length=True, padding=False, truncation=False)
        
        nl_lengths = nl_tokenized['length']
        sql_lengths = sql_tokenized['length']
        
        # Compute vocabulary sizes
        nl_vocab_size = compute_vocab_size(nl_texts, tokenizer)
        sql_vocab_size = compute_vocab_size(sql_texts, tokenizer)
        
        results[split] = {
            'num_examples': len(nl_texts),
            'mean_sentence_length': np.mean(nl_lengths),
            'mean_sql_length': np.mean(sql_lengths),
            'vocab_size_nl': nl_vocab_size,
            'vocab_size_sql': sql_vocab_size
        }
    
    # Print the table in the requested format
    print("\n" + "="*70)
    print("DATA STATISTICS (Using T5 Tokenizer)")
    print("="*70)
    print(f"\n{'Statistics Name':<40} {'Train':<15} {'Dev':<15}")
    print("-"*70)
    print(f"{'Number of examples':<40} {results['train']['num_examples']:<15} {results['dev']['num_examples']:<15}")
    print(f"{'Mean sentence length':<40} {results['train']['mean_sentence_length']:<15.2f} {results['dev']['mean_sentence_length']:<15.2f}")
    print(f"{'Mean SQL query length':<40} {results['train']['mean_sql_length']:<15.2f} {results['dev']['mean_sql_length']:<15.2f}")
    print(f"{'Vocabulary size (natural language)':<40} {results['train']['vocab_size_nl']:<15} {results['dev']['vocab_size_nl']:<15}")
    print(f"{'Vocabulary size (SQL)':<40} {results['train']['vocab_size_sql']:<15} {results['dev']['vocab_size_sql']:<15}")
    print("="*70)
    
    # Also print as markdown table for easy copy-paste
    print("\n\nMarkdown format (for easy copy-paste):")
    print("```")
    print(f"| Statistics Name | Train | Dev |")
    print(f"|----------------|-------|-----|")
    print(f"| Number of examples | {results['train']['num_examples']} | {results['dev']['num_examples']} |")
    print(f"| Mean sentence length | {results['train']['mean_sentence_length']:.2f} | {results['dev']['mean_sentence_length']:.2f} |")
    print(f"| Mean SQL query length | {results['train']['mean_sql_length']:.2f} | {results['dev']['mean_sql_length']:.2f} |")
    print(f"| Vocabulary size (natural language) | {results['train']['vocab_size_nl']} | {results['dev']['vocab_size_nl']} |")
    print(f"| Vocabulary size (SQL) | {results['train']['vocab_size_sql']} | {results['dev']['vocab_size_sql']} |")
    print("```")

if __name__ == "__main__":
    main()

