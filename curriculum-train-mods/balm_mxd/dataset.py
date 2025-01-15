import torch
from torch.utils.data import Dataset
import random
import pandas as pd
from balm_mxd import calc_prob
from datasets import load_dataset
from accelerate import Accelerator

# Samples based on the probability of unpaired data
# Probability is updated the MixedProbCallback (if not constant)
class MixedProbDataset(Dataset):
    def __init__(self, 
                 tokenized_data, # expects dict containing both 'unpaired' and 'paired' dataset (dataset can be 'None')
                 num_training_steps,
                 is_train=True,
                 constant_prob=False, # if unpaired prob should be constant
                 curr_prob={"k": 15, "shift": 0.5, "A": -1, "B": 1}, # if constant_prob=False, these are default values for the curr prob function
                 prob=None, # if constant_prob == True, use this probability (otherwise gets ignored)
                 seed=42,
                ):
        
        super().__init__()
        self.tokenized_data = tokenized_data
        self.is_train = is_train
        self.num_training_steps = num_training_steps
        self.constant_prob = constant_prob
        self.curr_prob = curr_prob

        # seperate unpaired and paired data based on 'label' column
        if is_train:
            # seperate unpaired and paired data
            self.unpaired_data = tokenized_data['unpaired']
            self.paired_data = tokenized_data['paired']

            # set initial probability
            self.current_prob = calc_prob(0, num_training_steps, curr_prob) if not constant_prob else prob
            
            # track unpaired and paired count for epoch calculations
            self.unpaired_count = 0
            self.paired_count = 0  

            # generator for random sampling
            seed = seed + Accelerator().process_index
            self.generator = torch.Generator().manual_seed(seed)

    def set_probability(self, probability):
        self.current_prob = probability

    # when training, ignores idx provided by dataloader (b/c it has no awareness of the two datasets) 
    # instead randomly selects idx from the dataset
    def __getitem__(self, idx):
        if self.is_train:
            if (self.current_prob == 0) or (random.random() > self.current_prob):
                self.paired_count += 1
                random_idx = torch.randint(high = len(self.paired_data), size=(1,), 
                                           generator=self.generator).item()
                return self.paired_data[random_idx]
            else:
                self.unpaired_count += 1
                random_idx = torch.randint(high = len(self.unpaired_data), size=(1,), 
                                           generator=self.generator).item()
                return self.unpaired_data[random_idx]

        # eval - use dataloader idx
        return self.tokenized_data[idx % len(self.tokenized_data)]

    def __len__(self):
        if self.is_train:
            # account for one of the datasets potentially being 'None'
            unpaired_len = len(self.unpaired_data) if self.unpaired_data is not None else 0
            paired_len = len(self.paired_data) if self.paired_data is not None else 0
            return (unpaired_len + paired_len)
        else:
            return len(self.tokenized_data)

def tokenize(
    seq, 
    tokenizer, 
    padding="max_length",
    truncation=True,
    max_len=320,
) -> list:
    
    tokenized = tokenizer(seq["sequence"], 
                          padding=padding, 
                          max_length=max_len,
                          truncation=truncation,
                          return_special_tokens_mask=True)
    
    return tokenized

def process_datasets(data_files, 
                     tokenizer, 
                     config,
                     constant_prob=False,
                     curr_prob={"k": 15, "shift": 0.5, "A": -1, "B": 1},
                     prob=None,
                     seed=42,
                     num_proc=128,
                     cache_dir="~/.cache/huggingface/datasets"):
    
    # check for None values and remove from dict
    none_keys = [k for k in data_files if data_files[k] is None]
    data_files = {k: v for k, v in data_files.items() if k not in none_keys}
    
    # load
    dataset = load_dataset("parquet", 
                           data_files=data_files, 
                           num_proc=num_proc,
                           cache_dir=cache_dir)

    # tokenize
    # be careful with this caching method 
    # if you change the tokenization, you should delete
    # the cache manually or it seems to reuse the previous cache
    tokenized_dataset = dataset.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "padding": config.get("padding"),
            "max_len": config.get("max_len"),
            "truncation": config.get("truncate"),
        },
        cache_file_names = {k: f'{cache_dir}/{str(k)}.arrow' for k in dataset},
        num_proc=num_proc,
        remove_columns=["sequence", "sequence_id"]
    )

    # add back None keys
    tokenized_dataset.update({key: None for key in none_keys})

    # format
    train_dataset = MixedProbDataset({"paired": tokenized_dataset['paired_train'],
                                      "unpaired": tokenized_dataset['unpaired_train']}, 
                                    is_train=True,
                                    num_training_steps=config.get('max_steps'),
                                    constant_prob=constant_prob,
                                    curr_prob=curr_prob,
                                    prob=prob,
                                    seed=seed,)
    eval_dataset = {
        "paired": tokenized_dataset['paired_eval'],
        "unpaired": tokenized_dataset['unpaired_eval'],
    }
    
    return train_dataset, eval_dataset