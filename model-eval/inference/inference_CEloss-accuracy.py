import argparse
import csv
import os
import pathlib

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score
from transformers import (
    DataCollatorForLanguageModeling,
    EsmForMaskedLM,
    EsmTokenizer,
    Trainer,
)

from curriculum_mods import (
    MixedConfig,
    define_args,
    tokenize,
)


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Convert logits and labels to PyTorch tensors for CEL calculation
    logits_tensor = torch.tensor(logits)
    labels_tensor = torch.tensor(labels)

    # Create a mask that excludes padding (-100) and sep token
    mask_mod = (labels_tensor != -100) & ~torch.isin(
        labels_tensor, torch.tensor([0, 31])
    )
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

    return {"accuracy": accuracy_mod, "CE_loss": cel_mod.item()}


def main():
    # read args
    args = parser()

    # load test data
    paired = pd.read_parquet(f"./paired_sep_test-annotated.parquet")[
        ["sequence_id", "sequence"]
    ]
    unpaired = pd.read_parquet(f"./unpaired_sep_test10k-annotated.parquet")[
        ["sequence_id", "sequence"]
    ]

    dataset = DatasetDict(
        {
            "test-paired": Dataset.from_pandas(paired),
            "test-unpaired": Dataset.from_pandas(unpaired),
        }
    )

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
        remove_columns=["sequence_id", "sequence"],
    )

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # load model
    model = EsmForMaskedLM.from_pretrained(args.model_path)

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

    # Prepend model name to results
    res = {"model_name": args.model_name, **res}

    # Check if header needs to be created in output file
    write_header = (
        not os.path.isfile(args.output_file) or os.stat(args.output_file).st_size == 0
    )

    with open(args.output_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=res.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(res)


if __name__ == "__main__":
    main()
