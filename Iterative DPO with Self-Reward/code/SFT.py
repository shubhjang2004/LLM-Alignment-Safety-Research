# SFT Training - Complete Code
# Run this entire notebook to train the SFT model

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import os

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Configuration
class Config:
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "./outputs"
    sft_model_dir = "./outputs/sft_model"
    dataset_name = "Anthropic/hh-rlhf"
    num_sft_samples = 10000
    max_length = 512
    sft_epochs = 3
    batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    use_wandb = False

config = Config()
os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.sft_model_dir, exist_ok=True)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load and prepare data
print("Loading HH-RLHF dataset...")
dataset = load_dataset(config.dataset_name, split="train")
helpful_data = dataset.filter(lambda x: len(x['chosen']) > 0)
helpful_data = helpful_data.select(range(min(config.num_sft_samples, len(helpful_data))))

def format_sft(example):
    try:
        prompt = example['chosen'].split('Assistant:')[0].replace('Human:', '').strip()
        response = example['chosen'].split('Assistant:')[-1].strip()
        return {"text": f"Human: {prompt}\n\nAssistant: {response}"}
    except:
        return {"text": ""}

sft_dataset = helpful_data.map(format_sft, remove_columns=helpful_data.column_names)
sft_dataset = sft_dataset.filter(lambda x: len(x['text']) > 0)
print(f"SFT dataset size: {len(sft_dataset)}")

# Tokenize data
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=config.max_length,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing data...")
tokenized_dataset = sft_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=sft_dataset.column_names,
    desc="Tokenizing"
)

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Apply LoRA
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
print("Trainable parameters:")
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir=config.sft_model_dir,
    num_train_epochs=config.sft_epochs,
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Train
print("Starting SFT training...")
trainer.train()

# Save model
print("Saving model...")
model.save_pretrained(config.sft_model_dir)
tokenizer.save_pretrained(config.sft_model_dir)
print(f"✓ SFT model saved to {config.sft_model_dir}")

# Test generation
print("\n" + "="*50)
print("Testing generation...")
model.eval()
test_prompt = "Human: What is machine learning?\n\nAssistant:"
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
print("="*50)
print("\n✓ SFT Training Complete!")