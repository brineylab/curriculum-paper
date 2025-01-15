from transformers import (
    EsmConfig,
    TrainingArguments,
)

MixedConfig = {
    "seed": 42,
    "fp16": True,
    
    # model architecture
    # 55M params
    "num_hidden_layers": 5,
    "num_attention_heads": 20,
    "hidden_size": 960, # biggest effect on # of parameters
    "intermediate_size": 3840, # 4x the hidden size
    "vocab_size": 33,
    "pad_token_id": 1,
    "mask_token_id": 32,
    "max_len": 320,
    "max_position_embeddings": 322,
    "position_embedding_type": "rotary",
    
    # tokenizer
    "padding": "max_length",
    "truncate": True,
    
    # training parameters
    "batch_size": 128,
    "max_steps": 500000,
    "warmup_steps": 30000,
    "weight_decay": 0.01,
    "peak_learning_rate": 4e-4,
    "lr_scheduler_type": "linear",
    "lr_scheduler_kwargs": {},
    "adam_epsilon": 1e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "gradient_accumulation_steps": 1,
    "remove_unused_columns": False,
    
    # eval
    "evaluation_strategy": "steps",
    "eval_steps": 25000,
    
    # outputs and logging
    "save_steps": 25000,
    "overwrite_output_dir": True,
    "logging_steps": 10,
    "logging_first_step": True,
    "report_to": "wandb"
}

def define_config(config):
    model_config = EsmConfig(
        vocab_size = config.get("vocab_size"),
        pad_token_id = config.get("pad_token_id"),
        mask_token_id = config.get("mask_token_id"),
        hidden_size = config.get("hidden_size"),
        intermediate_size = config.get("intermediate_size"),
        max_position_embeddings= config.get("max_position_embeddings"),
        num_hidden_layers = config.get("num_hidden_layers"),
        num_attention_heads = config.get("num_attention_heads"),
        position_embedding_type = config.get("position_embedding_type")
    )
    return model_config

def define_args(config, run_name):
    training_args = TrainingArguments(
        run_name = run_name,
        seed = config.get("seed"),
        fp16 = config.get("fp16"),
        
        # batch sizes
        per_device_train_batch_size = config.get("batch_size"),
        per_device_eval_batch_size = config.get("batch_size"),
        
        # steps
        max_steps = config.get("max_steps"),
        save_steps = config.get("save_steps"),
        logging_steps = config.get("logging_steps"),
        
        # eval
        evaluation_strategy = config.get("evaluation_strategy"),
        eval_steps = config.get("eval_steps"),
        
        # training
        adam_beta1 = config.get("adam_beta1"),
        adam_beta2 = config.get("adam_beta2"),
        adam_epsilon = config.get("adam_epsilon"),
        weight_decay = config.get("weight_decay"),
        warmup_steps = config.get("warmup_steps"),
        learning_rate = config.get("peak_learning_rate"),
        lr_scheduler_type = config.get("lr_scheduler_type"),
        lr_scheduler_kwargs = config.get("lr_scheduler_kwargs"),
        gradient_accumulation_steps = config.get("gradient_accumulation_steps"),
        remove_unused_columns = config.get("remove_unused_columns"),
        
        # output and logging
        output_dir = f"./checkpoints/{run_name}",
        overwrite_output_dir = config.get("overwrite_output_dir"),
        logging_dir = f"./wandb/{run_name}",
        report_to = config.get("report_to"),
        logging_first_step = config.get("logging_first_step"),
    )
    return training_args