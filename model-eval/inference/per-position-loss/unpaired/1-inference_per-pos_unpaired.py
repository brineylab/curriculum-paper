import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import (
    EsmTokenizer, 
    EsmForMaskedLM
)
import torch
import torch.nn.functional as F
from tqdm import tqdm
from itertools import chain
import argparse
import pathlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def inference_heavy(model, tokenizer, seq, sep):
    losses = []
    preds = []
    perplexities = []
    
    with torch.no_grad():
        unmasked = tokenizer(
                       (seq + sep),
                       return_tensors='pt'
                   ).to(device)['input_ids']

        
        for m in range(len(seq)):
            hmask = seq[:m] + '<mask>' + seq[m+1:] + sep
            i = tokenizer(
                    hmask,
                    return_tensors='pt'
                ).to(device)
            mask_pos = (i.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
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
            'actual': seq,
            'sep': sep,
            'prediction': preds,
            'loss': losses,
            'perplexity': perplexities
        }

def inference_light(model, tokenizer, seq, sep):
    losses = []
    preds = []
    perplexities = []
    
    with torch.no_grad():
        unmasked = tokenizer(
                       (sep + seq),
                       return_tensors='pt'
                   ).to(device)['input_ids']
        
        for m in range(len(seq)):
            hmask = sep + seq[:m] + '<mask>' + seq[m+1:]
            i = tokenizer(
                    hmask,
                    return_tensors='pt'
                ).to(device)
            mask_pos = (i.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
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
            'actual': seq,
            'sep': sep,
            'prediction': preds,
            'loss': losses,
            'perplexity': perplexities
        }


def main():
    args = parser()
    
    sep_str = "sep"
    sep = "<sep>"
    
    df = pd.read_csv('./data/TTE-ds/annotated/unpaired-1k-annotated.csv')
    tokenizer = EsmTokenizer.from_pretrained("../../../tokenizer/")
    model = EsmForMaskedLM.from_pretrained(args.model).to('cuda')
        
    unpaired_data = []
    paired_data = []
    for pair_id, row in tqdm(list(df.iterrows())):
        chain = row['chain_type']
        if chain == 'H':
            d = {'sequence_id': row['sequence_id'], 'locus': 'heavy'}
            d.update(inference_heavy(
                model, 
                tokenizer, 
                seq = row['sequence'],
                sep = sep,
            ))
            unpaired_data.append(d)
        else: # light chains
            d = {'sequence_id': row['sequence_id'], 'locus': 'light'}
            d.update(inference_light(
                model, 
                tokenizer, 
                seq = row['sequence'], 
                sep = sep,
            ))
            unpaired_data.append(d)

    # save
    unpaired_df = pl.DataFrame(unpaired_data)
    unpaired_df.write_parquet(
        f'./results/per-position/{args.model_name}_unpaired1k-perpos-loss.parquet',
    )

if __name__ == "__main__":
    main()