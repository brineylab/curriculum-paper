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
        default="paired-only_<cls>_lr1e-4_650M-ESM_500k-stp"
    )
    parser.add_argument(
        "--paired_dir",
        required=True,
    )
    parser.add_argument(
        "--unpaired_dir",
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        required=True,
    )
    args = parser.parse_args()
    return args
    
def main():
    args =  parser()
    run_name = f"{args.run_name}_{date.today().isoformat()}"
    
    # seed
    set_seed(MixedConfig.get('seed'))

    # update config for 650M model
    MixedConfig['num_hidden_layers'] = 33
    MixedConfig['hidden_size'] = 1280
    MixedConfig['intermediate_size'] = 5120
    MixedConfig['batch_size'] = 64
    MixedConfig['peak_learning_rate'] =  1e-4
    
    # load, tokenize, & format data
    tokenizer = EsmTokenizer.from_pretrained("../tokenizer/vocab.txt")
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
                                                   cache_dir=args.cache_dir)

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # wandb
    os.environ['WANDB_PROJECT'] = 'mxd-data_fx'
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
