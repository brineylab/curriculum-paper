import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import EsmTokenizer, EsmForMaskedLM
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import chain
import argparse
import pathlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
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
    args = parser.parse_args()
    return args


def inference_paired(model, tokenizer, h, l, sep):
    losses = []
    preds = []
    perplexities = []

    with torch.no_grad():
        unmasked = tokenizer(h + sep + l, return_tensors="pt").to(device)["input_ids"]

        # heavy chain
        for m in range(len(h)):
            hmask = h[:m] + "<mask>" + h[m + 1 :]
            i = tokenizer(hmask + sep + l, return_tensors="pt").to(device)
            mask_pos = (i.input_ids == tokenizer.mask_token_id)[0].nonzero(
                as_tuple=True
            )[0]
            labels = torch.where(i.input_ids == tokenizer.mask_token_id, unmasked, -100)
            output = model(**i, labels=labels)

            # PPL
            logits = output.logits
            ce_loss = output.loss
            perplexities.append(float(torch.exp(ce_loss)))

            # predictions
            pred_token = logits[0, mask_pos].argmax(axis=-1)
            preds.append(tokenizer.decode(pred_token))

            # loss
            loss = output.loss.item()
            losses.append(loss)

        # light chain
        for m in range(len(l)):
            lmask = l[:m] + "<mask>" + l[m + 1 :]
            i = tokenizer(h + sep + lmask, return_tensors="pt").to(device)
            mask_pos = (i.input_ids == tokenizer.mask_token_id)[0].nonzero(
                as_tuple=True
            )[0]
            labels = torch.where(i.input_ids == tokenizer.mask_token_id, unmasked, -100)
            output = model(**i, labels=labels)

            # PPL
            logits = output.logits
            ce_loss = output.loss
            perplexities.append(float(torch.exp(ce_loss)))

            # predictions
            pred_token = logits[0, mask_pos].argmax(axis=-1)
            preds.append(tokenizer.decode(pred_token))

            # loss
            loss = output.loss.item()
            losses.append(loss)

        return {
            "heavy": h,
            "sep": sep,
            "light": l,
            "actual": h + l,
            "prediction": preds,
            "loss": losses,
            "perplexity": perplexities,
        }


def main():
    args = parser()

    sep_str = "sep"
    sep = "<sep>"

    df = pd.read_csv("./data/TTE-ds/annotated/paired-1k-annotated.csv")
    tokenizer = EsmTokenizer.from_pretrained("../../../tokenizer/")
    model = EsmForMaskedLM.from_pretrained(args.model).to(device)

    unpaired_data = []
    paired_data = []
    for pair_id, row in tqdm(list(df.iterrows())):
        # paired
        d = {"sequence_id": row["sequence_id"]}
        d.update(
            inference_paired(
                model,
                tokenizer,
                h=row["sequence_aa_heavy"],
                l=row["sequence_aa_light"],
                sep=sep,
            )
        )
        paired_data.append(d)

    # save
    paired_df = pl.DataFrame(paired_data)
    paired_df.write_parquet(
        f"./results/per-position/{args.model_name}_paired1k-perpos-loss.parquet",
    )


if __name__ == "__main__":
    main()
