import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.split = split
        self.data_folder = data_folder
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # Load natural language queries and SQL queries
        nl_path = os.path.join(data_folder, f"{split}.nl")
        self.nl_queries = load_lines(nl_path)
        
        if split != "test":  # Test set doesn't have ground truth SQL
            sql_path = os.path.join(data_folder, f"{split}.sql")
            self.sql_queries = load_lines(sql_path)
        else:
            self.sql_queries = None
    
    def __len__(self):
        return len(self.nl_queries)

    def __getitem__(self, idx):
        nl_query = self.nl_queries[idx]
        
        # Tokenize encoder input (natural language)
        encoder_inputs = self.tokenizer(
            nl_query, 
            max_length=512, 
            padding="max_length", 
            truncation=True,
            return_tensors="pt"
        )
        
        if self.split != "test":
            sql_query = self.sql_queries[idx]
            # Tokenize decoder input and target (SQL)
            # T5 uses pad_token_id as the decoder start token
            # For teacher forcing: decoder_input = [pad, w1, w2, ..., wn-1]
            #                      decoder_target = [w1, w2, ..., wn, eos]
            decoder_inputs = self.tokenizer(
                sql_query,
                max_length=511,  # Leave room for pad_token_id at start
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Get the tokenized SQL sequence
            sql_ids = decoder_inputs["input_ids"].squeeze(0)
            
            # Decoder input: prepend pad_token_id, remove last token
            # This ensures decoder_input and decoder_target are same length
            decoder_input_ids = torch.cat([
                torch.tensor([self.tokenizer.pad_token_id]),
                sql_ids[:-1]
            ])
            
            # Decoder target: the actual SQL sequence (for loss computation)
            decoder_target_ids = sql_ids
            
            return {
                "encoder_ids": encoder_inputs["input_ids"].squeeze(0),
                "encoder_mask": encoder_inputs["attention_mask"].squeeze(0),
                "decoder_inputs": decoder_input_ids,
                "decoder_targets": decoder_target_ids,
                "initial_decoder_input": self.tokenizer.pad_token_id
            }
        else:
            return {
                "encoder_ids": encoder_inputs["input_ids"].squeeze(0),
                "encoder_mask": encoder_inputs["attention_mask"].squeeze(0),
                "initial_decoder_input": self.tokenizer.pad_token_id
            }

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item["encoder_ids"] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([item["encoder_mask"] for item in batch], batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence([item["decoder_inputs"] for item in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([item["decoder_targets"] for item in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.tensor([item["initial_decoder_input"] for item in batch])
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item["encoder_ids"] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([item["encoder_mask"] for item in batch], batch_first=True, padding_value=0)
    initial_decoder_inputs = torch.tensor([item["initial_decoder_input"] for item in batch])
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x