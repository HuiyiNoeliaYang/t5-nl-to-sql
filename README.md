# T5 Natural Language to SQL

A T5-based neural machine translation system that converts natural language queries into SQL statements for querying a flight information database.

## Project Description

This project implements a semantic parsing system that translates natural language questions about flight information into executable SQL queries. The system uses Google's T5 (Text-to-Text Transfer Transformer) model with optional fine-tuning capabilities and constrained decoding to ensure syntactically valid SQL generation.

### Example Translations

**Input:** "what flights are available tomorrow from denver to philadelphia"

**Output:**
```sql
SELECT DISTINCT flight_1.flight_id FROM flight flight_1, airport_service airport_service_1,
city city_1, airport_service airport_service_2, city city_2, days days_1, date_day date_day_1
WHERE flight_1.from_airport = airport_service_1.airport_code
AND airport_service_1.city_code = city_1.city_code
AND city_1.city_name = 'DENVER' ...
```

## Features

- **T5-based Translation**: Leverages pre-trained T5 models for sequence-to-sequence learning
- **Fine-tuning Options**: Supports various fine-tuning strategies including:
  - Full model fine-tuning
  - Encoder/decoder freezing
  - Layer-wise freezing
  - Embedding layer freezing
- **Constrained Decoding**: Ensures generated SQL queries follow valid SQL syntax
- **Multiple Training Strategies**: Configurable optimizer, learning rate scheduling, and early stopping
- **Comprehensive Evaluation**: F1-score based evaluation using both SQL string matching and database record comparison
- **WandB Integration**: Built-in Weights & Biases support for experiment tracking

## Installation

### Environment Setup

It's highly recommended to use a virtual environment (e.g., conda, venv) for this project.

Using conda:
```bash
conda create -n t5-nl-to-sql python=3.10
conda activate t5-nl-to-sql
pip install -r requirements.txt
```

Using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Training

Train the T5 model with fine-tuning:
```bash
python train_t5.py --finetune \
  --learning_rate 1e-4 \
  --max_n_epochs 10 \
  --patience_epochs 3
```

Or use the recommended training script:
```bash
bash train_recommended.sh
```

### Evaluation

Evaluate predictions using F1 scores:
```bash
python evaluate.py \
  --predicted_sql results/t5_ft_dev.sql \
  --predicted_records records/t5_ft_dev.pkl \
  --development_sql data/dev.sql \
  --development_records records/ground_truth_dev.pkl
```

### Testing Checkpoints

Evaluate saved model checkpoints:
```bash
python evaluate_checkpoint.py \
  --checkpoint_path checkpoints/model_epoch_5.pt \
  --data_split dev
```

## Project Structure

```
t5-nl-to-sql/
├── data/                       # Dataset files
│   ├── train.nl/train.sql     # Training data
│   ├── dev.nl/dev.sql         # Development/validation data
│   ├── test.nl                # Test queries
│   ├── flight_database.db     # SQLite database
│   └── flight_database.schema # Database schema
├── checkpoints/               # Saved model checkpoints
├── results/                   # Generated SQL queries
├── records/                   # Database query results
├── train_t5.py               # Main training script
├── evaluate.py               # Evaluation script
├── evaluate_checkpoint.py    # Checkpoint evaluation
├── constrained_decoding.py   # SQL syntax constraints
├── load_data.py              # Data loading utilities
├── t5_utils.py               # T5 model utilities
└── utils.py                  # General utilities
```

## Dataset

The project uses the ATIS (Airline Travel Information System) dataset, which contains:
- Natural language queries about flight information
- Corresponding SQL queries
- A flight information database with tables for flights, airports, cities, airlines, etc.

## Model Architecture

The system uses the T5 encoder-decoder architecture:
1. **Encoder**: Processes natural language input
2. **Decoder**: Generates SQL output token by token
3. **Constrained Decoding**: Applies SQL grammar constraints during generation

## Evaluation Metrics

The system is evaluated using F1 scores based on:
- **SQL String Matching**: Exact match of generated SQL queries
- **Database Record Matching**: Comparison of query execution results

## Documentation

Additional documentation is available:
- [CONSTRAINED_DECODING.md](CONSTRAINED_DECODING.md) - SQL constraint implementation details
- [GENERATION_FLOW.md](GENERATION_FLOW.md) - Text generation pipeline
- [GENERATION_PARAMETERS_EXPLAINED.md](GENERATION_PARAMETERS_EXPLAINED.md) - Generation parameter guide
- [MODEL_SAVING_EXPLAINED.md](MODEL_SAVING_EXPLAINED.md) - Model checkpoint management
- [EVALUATE_CHECKPOINT_GUIDE.md](EVALUATE_CHECKPOINT_GUIDE.md) - Checkpoint evaluation guide

## Requirements

- Python 3.10+
- PyTorch
- Transformers (HuggingFace)
- SQLite3
- WandB (optional, for experiment tracking)

See [requirements.txt](requirements.txt) for complete dependencies.

## License

This project is for educational and research purposes.
