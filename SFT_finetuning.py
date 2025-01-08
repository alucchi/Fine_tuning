#!/usr/bin/env python
# coding: utf-8

# # Same as before with LoRA
# !python -m pip install -U wandb transformers trl datasets "protobuf==3.20.3" evaluate peft


# Set NCCL environment variables (needed for some GPUs)
# import os
#os.environ["NCCL_P2P_DISABLE"] = "1"
#os.environ["NCCL_IB_DISABLE"] = "1"


################################################################################
# Parameters

gradient_accumulation_steps = 2
num_train_epochs = 2

# Optimizer
#optim_name = "sgd"
#optim_name = "adam"
#optim_name = "mars"
#optim_lr = 1e-3 # learning rate

# dataset used for fine tuning
dataset_name = "AI-MO/NuminaMath-CoT"
# dataset_name = "gsm8k"

# base model
# model_id = 'meta-llama/Llama-2-7b-hf'
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# lr_scheduler_type = "cosine"
lr_scheduler_type = "linear"
logging_steps = 10

import argparse

parser = argparse.ArgumentParser(description="Script parameters")
parser.add_argument("--batch_size", type=int, default=4,
                                    help="Batch size for training")
parser.add_argument("--optim_name", type=str, default="sgd",
                                    help="Name of the optimizer (e.g. sgd, adam, mars)")
parser.add_argument("--n_training", type=int, default=1000,
                                    help="Number of training examples to use")
parser.add_argument("--optim_lr", type=float, default=1e-3,
                    help="Learning rate for the optimizer")

args = parser.parse_args()

# Update parameters from arguments
batch_size = args.batch_size
optim_name = args.optim_name
n_training = args.n_training
optim_lr = args.optim_lr

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

# Name of the run in wandb
run_name = "llama_finetuning_" + dataset_name.split('/')[-1] + "_" + str(n_training) + "_" + optim_name + "_l" + lr_scheduler_type + str(optim_lr) + "_b" + str(batch_size)

# Directory where to output models and checkpoints
output_dir = "./llama_ft_" + model_id.split('/')[-1] + "_" + dataset_name.split('/')[-1]
print("output_dir", output_dir)

# Initialize wandb
import wandb
wandb.init(project="llama_ft", # the project I am working on
           job_type="train",
           tags=["hf_sft_lora", "llama"],
           name=run_name) # the Hyperparameters I want to keep track of
            

# Load the dataset
from datasets import load_dataset
dataset = load_dataset(dataset_name)

print('dataset size:', len(dataset["train"]), len(dataset["test"]))

train_dataset = dataset["train"].select(range(n_training))
eval_dataset = dataset["test"]

print('len(train_dataset) ', len(train_dataset))
total_num_steps = num_train_epochs * len(train_dataset) // (batch_size * gradient_accumulation_steps)
total_num_steps = int(total_num_steps)
print('total_num_steps ', total_num_steps)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_kwargs = dict(
    device_map={"" : 0},
    trust_remote_code=True,
    # low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    # use_flash_attention_2=True,
    use_cache=False,
)


from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=64,  # the rank of the LoRA matrices
    lora_alpha=16, # the weight
    lora_dropout=0.1, # dropout to add to the LoRA layers
    bias="none", # add bias to the nn.Linear layers?
    task_type="CAUSAL_LM",
    #target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
    target_modules="all-linear"
)



from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//2,
    bf16=True,
    learning_rate=optim_lr,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio = 0.1,
    max_steps=total_num_steps,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True, # setting this to False to see if don't get out of CUDA memory error
    gradient_checkpointing_kwargs=dict(use_reentrant=False),
    evaluation_strategy="steps",
    eval_steps=total_num_steps // num_train_epochs,
    # eval_steps=10,
    # logging strategies
    logging_strategy="steps",
    logging_steps=logging_steps,
    save_strategy="steps",
    save_steps=total_num_steps // num_train_epochs,
)


from transformers.integrations import WandbCallback
from transformers import GenerationConfig
from tqdm import tqdm


class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, max_new_tokens=max_tokens, log_model=True):
        super().__init__()
        #self._log_model = log_model
        # self.sample_dataset = test_dataset.select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens,
                                                           load_in_8bit=True
                                                          )

    def on_train_begin(self, args, state, control, **kwargs):
        # You can add any specific logic you need here or leave it as pass
        pass


from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configuration with 8-bit loading
model_kwargs = dict(
    device_map="auto",               # Adjusts device automatically
    load_in_8bit=True,               # Enables 4-bit precision loading
    trust_remote_code=True,
    use_cache=False
)

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set the pad_token to eos_token or add a new special token for padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Option 1: Use eos_token as pad_token
    # Alternatively, Option 2: Add a new special pad_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))  # Resize embeddings if new token is added

model.generation_config.pad_token_id = tokenizer.pad_token_id

# Set the padding side to 'left'                                                                                                                                                                                           
tokenizer.padding_side = 'left'

    
# Modified SFTTrainer call


# Wrap the model with the PEFT configuration
from peft import get_peft_model
model = get_peft_model(model, peft_config)

# Now create the optimizer with the updated model parameters

if optim_name == "sgd":

    print("Using SGD optimizer\n")
    
    from torch.optim import SGD
    optimizer = SGD(model.parameters(), lr=optim_lr, momentum=0.9)

elif optim_name == "adam":

    print("Using Adam optimizer\n")
    
    # Define optimizer
    from torch.optim import Adam as PyTorchAdam

    # Define the optimizer with specified parameters
    optimizer = PyTorchAdam(
        model.parameters(),
        lr=optim_lr,
        betas=(0.9, 0.999),  ## the default value
        eps=1e-08,
        weight_decay=1e-4
        # correct_bias=True,
    )
else:

    print("Using MARS optimizer\n")
    
    from mars import MARS

    optimizer = MARS(model.parameters(), lr=optim_lr, weight_decay=1e-4, optimize_1d=False)

    
# # Define a learning rate scheduler
# from transformers import get_scheduler
# scheduler = get_scheduler(
#     name="linear",  # You can choose from "linear", "cosine", "polynomial", etc.
#     optimizer=optimizer,
#     num_warmup_steps=0,  # Set the number of warm-up steps (optional)
#     num_training_steps=total_num_steps
# )

# If you want a personalized prompt
# def create_prompt(row):
#     return ("Below is an instruction that describes a task. "
#             "Write a response that appropriately completes the request.\n\n"
#             "### Instruction:\n{problem}\n\n### Response:\n{solution}").format_map(row)


trainer = SFTTrainer(
    model=model,  # Directly pass the initialized 8-bit model
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,  # Add tokenizer if not yet specified
    #packing=True,
    #max_seq_length=512,
    args=training_args,
    #formatting_func=create_prompt,
    peft_config=peft_config,
    #optimizers=(create_optimizer(model), None)  # Pass optimizer and optional scheduler
    optimizers=(optimizer, None)
)


# remove answers
def create_prompt_no_anwer(row):
    row["solution"] = ""
    return {"text": create_prompt(row)}

test_dataset = eval_dataset.map(create_prompt_no_anwer)


# Training



wandb_callback = LLMSampleCB(trainer, test_dataset, num_samples=10, max_new_tokens=256)
trainer.add_callback(wandb_callback)
trainer.train()
wandb.finish()



# # Sampling from the model
# from transformers import GenerationConfig

# # Define the sampling configuration
# gen_config = GenerationConfig.from_pretrained(model_id)
# gen_config.max_new_tokens = 300  # Set max tokens for the generation
# gen_config.temperature = 0.7     # Control randomness in sampling (lower is more deterministic)

# # Sample prompts to generate from
# sample_prompts = [
#     eval_dataset[0]["problem"]
# ]

# # Generate and print the output for each prompt
# for prompt in sample_prompts:
#     inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()  # Adjust for CUDA if needed
#     with torch.no_grad():
#         outputs = model.generate(inputs, generation_config=gen_config)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"Prompt: {prompt}\nResponse: {generated_text}\n{'-' * 50}")



# Save the model and tokenizer
print("Saving model to", output_dir)
trainer.model.save_pretrained(output_dir)
trainer.tokenizer.save_pretrained(output_dir)

# If using LoRA, save the PEFT model configuration
from peft import PeftModel
if isinstance(trainer.model, PeftModel):
    trainer.model.save_pretrained(output_dir, safe_serialization=True)  # safer serialization for larger models


wandb.finish()
