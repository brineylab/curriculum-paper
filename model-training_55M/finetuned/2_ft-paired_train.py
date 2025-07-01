import os
import warnings

warnings.simplefilter("ignore")
from transformers import (
    EsmTokenizer,
    EsmForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
)
from curriculum_mods import (
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
        "--paired_dir",
        required=True,
    )
    parser.add_argument(
        "--unpaired_dir",
        required=True,
    )
    parser.add_argument(
        "--shards_dir",  # name of directory containing unpaired dataset shards
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        required=True,
    )
    args = parser.parse_args()
    return args


def main():
    # run name
    args = parser()
    run_name = f"ft-paired_50M-ESM_38k-stp_{date.today().isoformat()}"

    # update config steps & warmup
    MixedConfig["max_steps"] = 37500
    MixedConfig["warmup_steps"] = 2250
    MixedConfig["eval_steps"] = 5000

    # seed
    set_seed(MixedConfig.get("seed"))

    # load, tokenize, & format data
    tokenizer = EsmTokenizer.from_pretrained("../tokenizer/")
    shards_dir = f"{args.unpaired_dir}{args.shards_dir}"
    data_files = {
        "paired_train": f"{args.paired_dir}paired-train_20241002.parquet",
        "unpaired_train": None,
        "paired_eval": f"{args.paired_dir}paired-eval_20241002.parquet",
        "unpaired_eval": f"{args.unpaired_dir}unpaired-eval_20241002.parquet",
    }
    train_dataset, eval_dataset = process_datasets(
        data_files=data_files,
        tokenizer=tokenizer,
        config=MixedConfig,
        constant_prob=True,
        prob=0,
        cache_dir=args.cache_dir,
    )

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # wandb
    os.environ["WANDB_PROJECT"] = "mxd-data_fx"
    os.environ["WANDB_RUN_GROUP"] = "ft-model"

    # model
    model = EsmForMaskedLM.from_pretrained(
        "./models/ft-unpaired_50M-ESM_62k-stp_2024-12-13/"
    )

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
