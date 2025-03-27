#!/usr/bin/env python
# coding: utf-8

#import os
#os.environ["NCCL_P2P_DISABLE"] = "1"
#os.environ["NCCL_IB_DISABLE"] = "1"

################################################################################
# Parameters
################################################################################

gradient_accumulation_steps = 6
num_train_epochs = 8

lr_scheduler_type = "linear"
logging_steps = 5

# Maximum sequence length for both inputs and labels
max_length = 1024

import argparse

parser = argparse.ArgumentParser(description="Script parameters")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for training")
parser.add_argument("--optim_name", type=str, default="adam",
                    help="Name of the optimizer (e.g. sgd, adam, mars)")
parser.add_argument("--n_training", type=int, default=-1,
                    help="Number of training examples to use")
parser.add_argument("--optim_lr", type=float, default=1e-3,
                    help="Learning rate for the optimizer")
parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                    help="Name of the base model")
parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500",
                    help="Name of dataset used to fine-tune")


args = parser.parse_args()

batch_size = args.batch_size
optim_name = args.optim_name
n_training = args.n_training
optim_lr = args.optim_lr
model_id = args.model_id
dataset_name = args.dataset_name

print("batch_size", batch_size)
print("gradient_accumulation_steps", gradient_accumulation_steps)
print("num_train_epochs", num_train_epochs)
print("optim_name", optim_name)
print("optim_lr", optim_lr)
print("dataset_name", dataset_name)
print("model_id", model_id)
print("n_training", n_training)
print("lr_scheduler_type", lr_scheduler_type)
print("logging_steps", logging_steps)

################################################################################
# Load data and model
################################################################################

project_name = "SFT_" + dataset_name.split('/')[-1]
run_name   = "SFT_" + model_id.split('/')[-1] + "_" + dataset_name.split('/')[-1] + "_n" + str(n_training) + "_o" + optim_name + "_l" + lr_scheduler_type + str(optim_lr) + "_b" + str(batch_size)
output_dir = "SFT_" + model_id.split('/')[-1] + "_" + dataset_name.split('/')[-1] + "_n" + str(n_training) + "_o" + optim_name + "_l" + lr_scheduler_type + str(optim_lr) + "_b" + str(batch_size)
print("project_name", project_name)
print("run_name", run_name)
print("output_dir", output_dir)

import wandb
wandb.init(project=project_name,
           job_type="train",
           tags=["hf_sft", "llama"],
           name=run_name)

from datasets import load_dataset

print("Loading training dataset")
if dataset_name == "gsm8k":
    train_dataset = load_dataset(dataset_name, "main")["train"]
    eval_dataset = None
    
elif dataset_name == "HuggingFaceH4/MATH-500":
    train_dataset = load_dataset(dataset_name, split="test")
    eval_dataset = None
else:
    print("Unkown dataset")
    raise SystemExit(1)


if n_training > 0:
    train_dataset = train_dataset.select(range(n_training))

print('len(train_dataset):', len(train_dataset))
total_num_steps = num_train_epochs * len(train_dataset) // (batch_size * gradient_accumulation_steps)
total_num_steps = int(total_num_steps)
print('total_num_steps:', total_num_steps)
#save_steps = total_num_steps // 2
#print('save_steps:', save_steps)

################################################################################
# Initialize model and tokenizer
################################################################################

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_kwargs = dict(
    device_map="auto",
    load_in_8bit=False,
    trust_remote_code=True,
    use_cache=False
)

model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.generation_config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = 'left'

################################################################################
# Prepare dataset with loss masking (mask prompt tokens)
################################################################################

# Define a system prompt that will be prepended to every example.
system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

def formatting_prompts_func(examples):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for question, cot, ans in zip(examples["problem"], examples["solution"], examples["answer"]):
        # Build prompt (input) and completion separately.
        prompt = system_prompt + "\n### Question:\n" + question + "\n### Response:\n"
        completion = cot + "\nThe final answer is \\boxed{" + ans + "}" + tokenizer.eos_token
        
        # Concatenate prompt and completion.
        full_text = prompt + completion
        
        # Tokenize the full text with attention mask; enforce max_length.
        tokenized_full = tokenizer(full_text, truncation=True, max_length=max_length, return_attention_mask=True)
        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        
        # Determine the length of the prompt (also truncated) to know where to mask.
        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=max_length)
        prompt_len = len(tokenized_prompt["input_ids"])
        
        # Create labels identical to input_ids...
        labels = input_ids.copy()
        # ...but mask out prompt tokens by setting them to -100.
        labels[:prompt_len] = [-100] * prompt_len
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
    
    return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": attention_mask_list}

# Apply the formatting function to the dataset.
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

################################################################################
# Custom Data Collator using pad_sequence
################################################################################

from torch.nn.utils.rnn import pad_sequence

def custom_data_collator(features):
    # For each key, convert the list of lists into a tensor by padding.
    batch = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        sequences = [torch.tensor(feature[key], dtype=torch.long) for feature in features]
        if key == "labels":
            padding_value = -100
        else:
            padding_value = tokenizer.pad_token_id
        batch[key] = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    return batch

################################################################################
# Set up training
################################################################################

from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size // 2,
    bf16=True,
    learning_rate=optim_lr,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=0.1,
    max_steps=total_num_steps,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs=dict(use_reentrant=False),
    evaluation_strategy="no",
    logging_strategy="steps",
    logging_steps=logging_steps,
    #save_strategy="steps",
    save_strategy="no",
    #save_steps=save_steps,
    max_grad_norm=1.0,
)

# Create the optimizer based on the provided name.
if optim_name == "sgd":
    print("Using SGD optimizer")
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=optim_lr, momentum=0.9)
elif optim_name == "adam":
    print("Using Adam optimizer")
    from torch.optim import Adam as PyTorchAdam
    optimizer = PyTorchAdam(model.parameters(), lr=optim_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
elif optim_name == "myadam":
    from myadam import MyAdam
    print("Using MyAdam optimizer")
    optimizer = MyAdam(model.parameters(), lr=optim_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
elif optim_name == "memadam":
    from memadam import Float16Adam
    print("Using SignAdam optimizer")
    optimizer = Float16Adam(model.parameters(), lr=optim_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
else:
    print("Using MARS optimizer")
    from mars import MARS
    optimizer = MARS(model.parameters(), lr=optim_lr, weight_decay=1e-4, optimize_1d=False)

# Sample callback for wandb logging (optional)
from transformers.integrations import WandbCallback
class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, max_new_tokens=256, log_model=True):
        super().__init__()
    def on_train_begin(self, args, state, control, **kwargs):
        pass

# Instantiate the SFTTrainer with the optimizer and custom data collator.
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    optimizers=(optimizer, None),
    data_collator=custom_data_collator,
)

wandb_callback = LLMSampleCB(trainer)
trainer.add_callback(wandb_callback)

################################################################################
# Train and Save
################################################################################

trainer.train()
wandb.finish()

print("Saving model to", output_dir)
trainer.model.save_pretrained(output_dir)
trainer.tokenizer.save_pretrained(output_dir)
wandb.finish()
