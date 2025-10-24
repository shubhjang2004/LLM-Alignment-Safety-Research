# DPO Training Round 1 - Fixed Version for Colab
# Run this AFTER generating synthetic preferences

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
import json
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Configuration
class Config:
    sft_model_dir = "./outputs/sft_model"
    dpo_round1_dir = "./outputs/dpo_round1"
    synthetic_data_path = "./outputs/synthetic_preferences.json"
    dataset_name = "Anthropic/hh-rlhf"
    max_length = 512
    max_prompt_length = 256
    dpo_epochs = 3
    batch_size = 1  # Reduced for memory
    gradient_accumulation_steps = 16  # Increased to compensate
    learning_rate = 5e-5
    beta = 0.1
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

config = Config()
os.makedirs(config.dpo_round1_dir, exist_ok=True)



# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.sft_model_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # DPO requires right padding

# Load data - combine original + synthetic
print("Loading preference data...")

# Original preferences
original_dataset = load_dataset(config.dataset_name, split="train")
original_dataset = original_dataset.select(range(min(3000, len(original_dataset))))

original_prefs = []
for ex in original_dataset:
    try:
        prompt = ex['chosen'].split('Assistant:')[0].replace('Human:', '').strip()
        chosen = ex['chosen'].split('Assistant:')[-1].strip()
        rejected = ex['rejected'].split('Assistant:')[-1].strip()
        if prompt and chosen and rejected:
            original_prefs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    except:
        continue

# Synthetic preferences
with open(config.synthetic_data_path, 'r') as f:
    synthetic_prefs = json.load(f)

# Remove score fields that DPO doesn't expect
synthetic_prefs_cleaned = []
for pref in synthetic_prefs:
    synthetic_prefs_cleaned.append({
        "prompt": pref["prompt"],
        "chosen": pref["chosen"],
        "rejected": pref["rejected"]
    })

# Combine
all_prefs = original_prefs + synthetic_prefs_cleaned
print(f"Total preferences: {len(all_prefs)} (Original: {len(original_prefs)}, Synthetic: {len(synthetic_prefs_cleaned)})")

# Create dataset
dpo_dataset = Dataset.from_list(all_prefs)

# Split into train/eval
train_test_split = dpo_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")


# Load model with fp16 (not bf16 - Colab T4 doesn't support bf16)
print("Loading SFT model...")
model = AutoModelForCausalLM.from_pretrained(
    config.sft_model_dir,
    torch_dtype=torch.float16,  # Use fp16 for Colab T4 compatibility
    device_map="auto",
    low_cpu_mem_usage=True
)

# Enable gradient checkpointing before applying LoRA
model.gradient_checkpointing_enable()

# Apply LoRA to main model
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

# DPO Training arguments
training_args = DPOConfig(
    output_dir=config.dpo_round1_dir,
    num_train_epochs=config.dpo_epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,  # Use fp16 instead of bf16 for Colab
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    beta=config.beta,
    max_length=config.max_length,
    max_prompt_length=config.max_prompt_length,
    remove_unused_columns=False,
    report_to="none",
    loss_type="sigmoid",
    # Memory optimization
    optim="adamw_torch",
    max_grad_norm=1.0,
)

# Create DPO trainer
print("Creating DPO trainer...")
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Let DPOTrainer create it from the base model
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  # ← CHANGED: Use processing_class instead of tokenizer
    # peft_config is already applied to model, so remove it from here
)

# Train
print("\nStarting DPO Round 1 training...")
print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
dpo_trainer.train()

# Save LoRA adapters
print("\nSaving DPO Round 1 LoRA adapters...")
model.save_pretrained(config.dpo_round1_dir)
tokenizer.save_pretrained(config.dpo_round1_dir)
print(f"✓ LoRA adapters saved to {config.dpo_round1_dir}")

# Test generation with LoRA
print("\n" + "="*50)
print("Testing generation with LoRA...")
model.eval()

test_prompt = "Human: How can I improve my productivity?\n\nAssistant:"
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
print("="*50)

# Optionally save merged model (commented out to save time/space)
# Uncomment if you want a standalone model without needing to load LoRA adapters
"""
print("\nMerging and saving full model...")
model = model.merge_and_unload()
merged_dir = config.dpo_round1_dir + "_merged"
os.makedirs(merged_dir, exist_ok=True)
model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print(f"✓ Merged model saved to {merged_dir}")
"""

print("\n✓ DPO Round 1 Training Complete!")
print(f"\nTo load this model later:")
print(f"from peft import PeftModel")
print(f"base_model = AutoModelForCausalLM.from_pretrained('{config.sft_model_dir}')")
print(f"model = PeftModel.from_pretrained(base_model, '{config.dpo_round1_dir}')")