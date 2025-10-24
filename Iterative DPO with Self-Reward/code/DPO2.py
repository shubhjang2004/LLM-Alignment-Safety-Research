# DPO Training Round 2 - Fixed for Colab
# Run this AFTER generating Round 2 synthetic preferences

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import json
import os

torch.manual_seed(43)
np.random.seed(43)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Configuration
class Config:
    sft_model_dir = "./outputs/sft_model"
    dpo_round1_dir = "./outputs/dpo_round1"
    dpo_round2_dir = "./outputs/dpo_round2"
    synthetic_data_path1 = "./outputs/synthetic_preferences.json"
    synthetic_data_path2 = "./outputs/synthetic_preferences_round2.json"
    dataset_name = "Anthropic/hh-rlhf"
    max_length = 512
    max_prompt_length = 256
    dpo_epochs = 2  # Reduced from 3 - model is already better from Round 1
    batch_size = 1
    gradient_accumulation_steps = 16
    learning_rate = 3e-5  # Lower LR for fine-tuning already trained model
    beta = 0.1
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

config = Config()



os.makedirs(config.dpo_round2_dir, exist_ok=True)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.dpo_round1_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # DPO requires right padding

# Load data - combine original + BOTH synthetic rounds
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

# Synthetic preferences Round 1
with open(config.synthetic_data_path1, 'r') as f:
    synthetic_prefs1 = json.load(f)

# Synthetic preferences Round 2
with open(config.synthetic_data_path2, 'r') as f:
    synthetic_prefs2 = json.load(f)

# Clean synthetic data (remove score fields)
synthetic_prefs1_cleaned = [
    {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
    for p in synthetic_prefs1
]
synthetic_prefs2_cleaned = [
    {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
    for p in synthetic_prefs2
]

# Combine all data
all_prefs = original_prefs + synthetic_prefs1_cleaned + synthetic_prefs2_cleaned
print(f"Total preferences: {len(all_prefs)}")
print(f"  - Original: {len(original_prefs)}")
print(f"  - Synthetic Round 1: {len(synthetic_prefs1_cleaned)}")
print(f"  - Synthetic Round 2: {len(synthetic_prefs2_cleaned)}")

# Create dataset
dpo_dataset = Dataset.from_list(all_prefs)

# Split into train/eval
train_test_split = dpo_dataset.train_test_split(test_size=0.1, seed=43)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

# Load DPO Round 1 model (Base + LoRA)
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    config.sft_model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("Loading DPO Round 1 LoRA adapters...")
model = PeftModel.from_pretrained(base_model, config.dpo_round1_dir)

# Merge LoRA weights into base model for Round 2 training
print("Merging LoRA adapters...")
model = model.merge_and_unload()

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Apply NEW LoRA for Round 2
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
    output_dir=config.dpo_round2_dir,
    num_train_epochs=config.dpo_epochs,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=30,  # Reduced warmup for fine-tuning
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    beta=config.beta,
    max_length=config.max_length,
    max_prompt_length=config.max_prompt_length,
    remove_unused_columns=False,
    report_to="none",
    loss_type="sigmoid",
    optim="adamw_torch",
    max_grad_norm=1.0,
)

# Create DPO trainer
print("Creating DPO trainer for Round 2...")
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Let DPOTrainer create reference model
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer

)

# Train
print("\nStarting DPO Round 2 training...")
print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
dpo_trainer.train()

# Save LoRA adapters
print("\nSaving DPO Round 2 LoRA adapters...")
model.save_pretrained(config.dpo_round2_dir)
tokenizer.save_pretrained(config.dpo_round2_dir)
print(f"✓ LoRA adapters saved to {config.dpo_round2_dir}")

# Test generation
print("\n" + "="*50)
print("Testing DPO Round 2 generation...")
model.eval()

test_prompts = [
    "Human: How can I improve my productivity?\n\nAssistant:",
    "Human: What is the meaning of life?\n\nAssistant:",
    "Human: Explain quantum computing simply.\n\nAssistant:"
]

for test_prompt in test_prompts:
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
    print("-"*50)

print("="*50)

print("\n✓ DPO Round 2 Training Complete!")
print("\n FULL ITERATIVE DPO PIPELINE COMPLETE!")

print(f"\nTo load Round 2 model later:")
print(f"from peft import PeftModel")
print(f"base_model = AutoModelForCausalLM.from_pretrained('{config.sft_model_dir}')")
print(f"model = PeftModel.from_pretrained(base_model, '{config.dpo_round2_dir}')")