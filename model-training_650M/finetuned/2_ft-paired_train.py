import os
import warnings
warnings.simplefilter('ignore')
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
    process_datasets,
    set_seed,
)
from datetime import date
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        default="ft-paired_<cls>_lr1e-4_650M-ESM_188k-stp",
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
    # run name
    args = parser()
    run_name = f"{args.run_name}_{date.today().isoformat()}"

    # update config steps & warmup
    MixedConfig['max_steps'] = 187500
    MixedConfig['warmup_steps'] = 11250
    MixedConfig["eval_steps"] = 5000

    # Model size - 650M
    MixedConfig['num_hidden_layers'] = 33
    MixedConfig['hidden_size'] = 1280
    MixedConfig['intermediate_size'] = 5120

    # Training Params
    MixedConfig['batch_size'] = 64
    MixedConfig['peak_learning_rate'] =  1e-4

    # seed
    set_seed(MixedConfig.get('seed'))
    
    # load, tokenize, & format data
    tokenizer = EsmTokenizer.from_pretrained("../tokenizer/")
    data_files = {
        "paired_train": f'{args.paired_dir}paired-train_20241119.parquet',
        "unpaired_train": None,
        "paired_eval": f'{args.paired_dir}paired-eval_20241119.parquet',
        "unpaired_eval": f'{args.unpaired_dir}unpaired-eval_20241119.parquet',
    }
    train_dataset, eval_dataset = process_datasets(data_files=data_files,
                                                   tokenizer=tokenizer,
                                                   config=MixedConfig,
                                                   constant_prob=True,
                                                   prob=0,
                                                   cache_dir=args.cache_dir,
                                                   seed=MixedConfig.get('seed'))

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # wandb
    os.environ['WANDB_PROJECT'] = 'mxd-data_fx'
    os.environ['WANDB_RUN_GROUP'] = 'large-scale'
    
    # model
    model = EsmForMaskedLM.from_pretrained('./models/ft-unpaired_<cls>_lr1e-4_650M-ESM_312k-stp/')

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
