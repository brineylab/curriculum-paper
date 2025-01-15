from transformers import TrainerCallback
import wandb
from accelerate import Accelerator
import math
import torch

def calc_prob(current_step, train_steps, vals):
    t = current_step / train_steps
    prob = 1 / (1 + math.exp(-vals['k'] * (t - vals['shift'])))
    prob = (vals['A'] * prob) + vals['B']
    return prob

class MixedProbCallback(TrainerCallback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.num_training_steps = dataset.num_training_steps
        self.accelerator = Accelerator()
        self.is_main_process = self.accelerator.is_main_process
        
        # for tracking eval losses
        self.eval_paired = None
        self.eval_unpaired = None

    # log initial values
    def on_train_begin(self, args, state, control, **kwargs):
        if self.is_main_process:
            wandb.log({
                "train/unpaired_epoch": 0,
                "train/paired_epoch": 0,
                "train/unpaired_probability": self.dataset.current_prob,
            }, step=0)
    
    # update probability based on current train step
    def on_step_begin(self, args, state, control, **kwargs):
        if not self.dataset.constant_prob:
            prob = calc_prob(state.global_step, self.num_training_steps, self.dataset.curr_prob)
            self.dataset.set_probability(prob)
    
    # log paired / unpaired epochs and unpaired probability
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step % state.logging_steps == 0:
            # aggregate counts across GPUs
            unpaired_count_tensor = torch.tensor([self.dataset.unpaired_count], dtype=torch.int64, device=self.accelerator.device)
            paired_count_tensor = torch.tensor([self.dataset.paired_count], dtype=torch.int64, device=self.accelerator.device)

            # accelerate reduce (sum)
            unpaired_count_tensor = self.accelerator.reduce(unpaired_count_tensor, reduction="sum")
            paired_count_tensor = self.accelerator.reduce(paired_count_tensor, reduction="sum")

            # calculate epoch fractions
            unpaired_epoch = unpaired_count_tensor.item() / len(self.dataset.unpaired_data) if self.dataset.unpaired_data != None else 0
            paired_epoch = paired_count_tensor.item() / len(self.dataset.paired_data) if self.dataset.paired_data != None else 0
            
            if self.is_main_process:
                wandb.log({
                    "train/unpaired_epoch": unpaired_epoch,
                    "train/paired_epoch": paired_epoch,
                    "train/unpaired_probability": self.dataset.current_prob,
                }, step=step)

    # log combined eval losses
    def on_evaluate(self, args, state, control, **kwargs):
        # extract losses
        metrics = kwargs['metrics']
        if 'eval_unpaired_loss' in metrics.keys():
            self.eval_unpaired = metrics['eval_unpaired_loss']
        elif 'eval_paired_loss' in metrics.keys():
            self.eval_paired = metrics['eval_paired_loss']
        
        # check if both paired and unpaired losses have been calculated
        if self.eval_unpaired is not None and self.eval_paired is not None:
            eval_average_loss = (self.eval_unpaired + self.eval_paired) / 2
            eval_weighted_loss = (self.eval_unpaired * self.dataset.current_prob) + (self.eval_paired * (1-self.dataset.current_prob))
            
            if self.is_main_process:
                wandb.log({
                    "eval/eval_average_loss": eval_average_loss,
                    "eval/eval_weighted_loss": eval_weighted_loss,
                })

            # reset before next eval step
            self.eval_unpaired, self.eval_paired = None, None