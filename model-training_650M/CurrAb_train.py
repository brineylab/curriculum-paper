import os
import warnings
warnings.simplefilter('ignore')
from transformers import (
    EsmTokenizer, 
    EsmForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
)
from balm_mxd import (
    MixedProbCallback,
    MixedConfig,
    define_args,
    define_config,
    process_datasets,
    set_seed,
)
from datetime import date
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        default="mxd-curr_max07-k15_<cls>_lr1e-4_650M-ESM_500k-stp",
    )
    # curriculum params
    parser.add_argument(
        "--k",
        default=15,
        type=float,
    )
    parser.add_argument(
        "--shift",
        default=0.8166,
        type=float,
    )
    parser.add_argument(
        "--A",
        default=-0.4,
        type=float,
    )
    parser.add_argument(
        "--B",
        default=0.7,
        type=float,
    )
    # data
    parser.add_argument(
        "--paired_dir",
        required=True,
    )
    parser.add_argument(
        "--unpaired_dir",
        required=True,
    )
    parser.add_argument(
        "--shards_dir", # name of directory containing unpaired dataset shards
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        required=True,
    )
    args = parser.parse_args()
    return args

def main():
    args = parser()
    run_name = f"{args.run_name}_{date.today().isoformat()}"

    # update config for 650M model
    MixedConfig['num_hidden_layers'] = 33
    MixedConfig['hidden_size'] = 1280
    MixedConfig['intermediate_size'] = 5120
    MixedConfig['batch_size'] = 64
    MixedConfig['peak_learning_rate'] =  1e-4

    # seed
    set_seed(MixedConfig.get('seed'))
    
    # load, tokenize, & format data
    tokenizer = EsmTokenizer.from_pretrained("../tokenizer/vocab.txt")
    shards_dir = f'{args.unpaired_dir}{args.shards_dir}'
    data_files = {
        "paired_train": f'{args.paired_dir}paired-train_20241119.parquet',
        "unpaired_train": [os.path.join(shards_dir, f) for f in os.listdir(shards_dir) if f.endswith('.parquet')],
        "paired_eval": f'{args.paired_dir}paired-eval_20241119.parquet',
        "unpaired_eval": f'{args.unpaired_dir}unpaired-eval_20241119.parquet',
    }
    train_dataset, eval_dataset = process_datasets(data_files=data_files,
                                                   tokenizer=tokenizer,
                                                   config=MixedConfig,
                                                   constant_prob=False,
                                                   curr_prob={"k": args.k, 
                                                              "shift": args.shift, 
                                                              "A": args.A, 
                                                              "B": args.B},
                                                   cache_dir=args.cache_dir,
                                                   seed=MixedConfig.get('seed'))

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # wand
    os.environ['WANDB_PROJECT'] = 'mxd-data'
    os.environ['WANDB_RUN_GROUP'] = 'large-scale'

    # model
    model_config = define_config(MixedConfig)
    model = EsmForMaskedLM(model_config)

    # training args
    training_args = define_args(MixedConfig, run_name)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[MixedProbCallback(train_dataset)],
    )

    # train
    trainer.train()
    trainer.save_model(f"./models/{run_name}")

if __name__ == "__main__":
    main()
