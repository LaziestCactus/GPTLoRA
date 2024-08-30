from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer
from loadData import LoRAdata
from LoRAdataset import PreprocessedDataset
from torch.amp import autocast
import torch
import os
import sys

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # Default to GPT small
GPTmodel = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank of the low-rank adaptation matrices
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.1,  # Dropout for LoRA layers
)

# Prepare model for LoRA tuning
model = get_peft_model(GPTmodel, lora_config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

#get Training dataset
dataname = 'tinyshakespeare.txt'
Tokenized_data = LoRAdata(dataname)

train_key, train_label = Tokenized_data.getTrain()
val_key, val_label = Tokenized_data.getVal()

train_dataset = PreprocessedDataset(train_key, train_label)
val_dataset = PreprocessedDataset(val_key, val_label)

# Define training arguments
num_epochs = 1  # Number of training epochs

training_args = TrainingArguments(
    output_dir='./results',        # Directory to save model checkpoints
    num_train_epochs=num_epochs,            # Number of training epochs
    per_device_train_batch_size=32, # Batch size per device
    per_device_eval_batch_size=32,  # Batch size for evaluation
    warmup_steps=500,              # Number of warmup steps
    weight_decay=0.01,             # Weight decay
    logging_dir='./logs',          # Directory to save logs
    logging_steps=10,              # Log every X steps
)

# Create Trainer instance
trainer = Trainer(
    model=model,                     # The model you are fine-tuning
    args=training_args,              # Training arguments
    train_dataset=train_dataset,     # Your training dataset
    eval_dataset=val_dataset,       # Your evaluation dataset (optional)
)





# Get model sizes
def print_model_size(path):
    size = 0
    for f in os.scandir(path):
        size += os.path.getsize(f)
    print(f"Model size: {(size / 1e6):.2} MB")

def print_trainable_parameters(model, label):
    parameters, trainable = 0, 0    
    for _, p in model.named_parameters():
        parameters += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    print(f"{label} trainable parameters: {trainable:,}/{parameters:,} ({100 * trainable / parameters:.2f}%)")





#Fine-tune the model
print(f"Model is on device: {next(model.parameters()).device}")
print_model_size(training_args.output_dir)
print_trainable_parameters(model, "Before training")
trainer.train()