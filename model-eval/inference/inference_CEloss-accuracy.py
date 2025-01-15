from dataclasses import dataclass
import numpy as np
from transformers import EsmForMaskedLM, EsmTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import DatasetDict, Dataset
import torch
import argparse
import pathlib
from tqdm import tqdm
import pickle
import csv
from balm_mxd import (
    MixedConfig,
    define_args,
    tokenize,
)
import pandas as pd
from sklearn.metrics import accuracy_score

def parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model", # model path
        default=None,
        required=True,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--model_name",
        default=None,
        required=True,
        type=str,
    )
    parser.add_argument(
        "--output_file",
        default=None,
        required=True,
        type=str,
    )
    args = parser.parse_args()
    return args


@dataclass
class ModelOutput:
    name: str
    chain: str
    states: np.ndarray

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Convert logits and labels to PyTorch tensors for CEL calculation
    logits_tensor = torch.tensor(logits)
    labels_tensor = torch.tensor(labels)

    # Create a mask that excludes padding (-100) and sep token
    mask_mod = (labels_tensor != -100) & ~torch.isin(labels_tensor, torch.tensor([0, 31]))
    labels_tensor_mod = labels_tensor[mask_mod]
    logits_tensor_mod = logits_tensor[mask_mod]

    # Calculate CEL with modified masked
    cel_mod = torch.nn.functional.cross_entropy(
        logits_tensor_mod, labels_tensor_mod, reduction="mean"
    )

    # Calculate accuracy
    predictions_mod = predictions[mask_mod.numpy()]
    labels_mod = labels_tensor_mod.numpy()
    accuracy_mod = accuracy_score(labels_mod, predictions_mod)
    
    return {
        "accuracy_mod": accuracy_mod,
        "CEL_mod": cel_mod.item()
    }

def main():
    # read args
    args = parser()

    # load test data
    path = "/home/jovyan/shared/Sarah/current/mixed-data_final"
    paired = pd.read_parquet(f'{path}/2_sep-tokens/eval/data/paired-test/paired_sep_test-annotated.parquet')[['sequence_id', 'sequence']]
    unpaired = pd.read_parquet(f'{path}/2_sep-tokens/eval/data/unpaired-test/unpaired_sep_test10k-annotated.parquet')[['sequence_id', 'sequence']]

    dataset = DatasetDict({
        "test-paired": Dataset.from_pandas(paired),
        "test-unpaired": Dataset.from_pandas(unpaired),
    })

    # tokenize
    tokenizer = EsmTokenizer.from_pretrained("../../tokenizer/")
    tokenized_dataset = dataset.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "padding": MixedConfig.get("padding"),
            "max_len": MixedConfig.get("max_len"),
            "truncation": True,
        },
        num_proc=16,
        remove_columns=['sequence_id', 'sequence']
    )

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # load model
    model = EsmForMaskedLM.from_pretrained(args.model)

    # inference
    MixedConfig["report_to"] = None
    training_args = define_args(MixedConfig, "eval")
    trainer = Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        compute_metrics=compute_metrics,
    )
    res = trainer.evaluate(tokenized_dataset)
    
    # CEL mod
    unpaired_CEL_mod = res['eval_test-unpaired_CEL_mod']
    paired_CEL_mod = res['eval_test-paired_CEL_mod']

    # accuracy mod
    unpaired_acc_mod = res['eval_test-unpaired_accuracy_mod']
    paired_acc_mod = res['eval_test-paired_accuracy_mod']
    
    with open(args.output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.model_name, unpaired_CEL_mod, unpaired_acc_mod, paired_CEL_mod, paired_acc_mod])

if __name__ == "__main__":
    main()