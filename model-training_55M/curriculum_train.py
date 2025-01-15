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
        required=True,
    )
    parser.add_argument(
        "--k",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--shift",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--A",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--B",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--paired_dir",
        default="/home/jovyan/shared/Sarah/current/mixed-data_final/2_sep-tokens/paired/data/sep/",
    )
    parser.add_argument(
        "--unpaired_dir",
        default="../mxd-sep-tokens/unpaired/data/sep/",
    )
    parser.add_argument(
        "--shards_dir",
        default="unpaired-train_20241002_shards/",
    )
    parser.add_argument(
        "--cache_dir",
        default=".cache/"
    )
    args = parser.parse_args()
    return args

def main():
    args = parser()
    run_name = f"{args.run_name}_{date.today().isoformat()}"

    # update config for shorter training time
    MixedConfig["max_steps"] = 100000
    MixedConfig["warmup_steps"] = 6000
    MixedConfig["eval_steps"] = 5000

    # seed
    set_seed(MixedConfig.get('seed'))
    
    # load, tokenize, & format data
    tokenizer = EsmTokenizer.from_pretrained("../tokenizer/")
    shards_dir = f'{args.unpaired_dir}{args.shards_dir}'
    data_files = {
        "paired_train": f'{args.paired_dir}paired-train_20241002.parquet',
        "unpaired_train": [os.path.join(shards_dir, f) for f in os.listdir(shards_dir) if f.endswith('.parquet')],
        "paired_eval": f'{args.paired_dir}paired-eval_20241002.parquet',
        "unpaired_eval": f'{args.unpaired_dir}unpaired-eval_20241002.parquet',
    }
    train_dataset, eval_dataset = process_datasets(data_files=data_files,
                                                   tokenizer=tokenizer,
                                                   config=MixedConfig,
                                                   constant_prob=False,
                                                   curr_prob={"k": args.k, 
                                                              "shift": args.shift, 
                                                              "A": args.A, 
                                                              "B": args.B},
                                                   cache_dir=args.cache_dir)

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # wand
    os.environ['WANDB_PROJECT'] = 'mxd-data_fx'
    os.environ['WANDB_RUN_GROUP'] = 'prob-curves'

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
