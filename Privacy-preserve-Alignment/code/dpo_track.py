

# Mount Google Drive
from google.colab import drive
import os

drive.mount('/content/drive')

if os.path.exists('/content/drive/MyDrive'):
    print(" Google Drive mounted successfully!")
else:
    print(" Drive mount failed!")

# Setup
import sys
import torch
import json
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import DPOTrainer, DPOConfig
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import time
from tqdm.auto import tqdm
import shutil

print(" Imports complete")

# Configure Paths (Drive + Local)
# === GOOGLE DRIVE PATHS (PERSISTENT) ===
DRIVE_BASE = Path("/content/drive/MyDrive/Project4_Privacy_Alignment")
DRIVE_DATA_DIR = DRIVE_BASE / "data"
DRIVE_MODELS_DIR = DRIVE_BASE / "models"
DRIVE_RESULTS_DIR = DRIVE_BASE / "results"

# === LOCAL PATHS (TEMPORARY - FASTER FOR TRAINING) ===
LOCAL_BASE = Path("/content")
LOCAL_DATA_DIR = LOCAL_BASE / "data"
LOCAL_MODELS_DIR = LOCAL_BASE / "models"
LOCAL_RESULTS_DIR = LOCAL_BASE / "results"
CHECKPOINT_DIR = LOCAL_BASE / "checkpoints"

# Create directories
for dir_path in [LOCAL_DATA_DIR, LOCAL_MODELS_DIR, LOCAL_RESULTS_DIR, 
                 CHECKPOINT_DIR, DRIVE_MODELS_DIR, DRIVE_RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

print(" Directories configured")
print(f" Data will load from: {DRIVE_DATA_DIR}")
print(f" Models will save to: {DRIVE_MODELS_DIR}")

# Load Data from Drive
print("\n Loading processed data from Google Drive...")

# Copy from Drive to local (faster for training)
drive_dataset_path = DRIVE_DATA_DIR / "hh_rlhf_processed"
local_dataset_path = LOCAL_DATA_DIR / "hh_rlhf_processed"

if not drive_dataset_path.exists():
    raise FileNotFoundError(
        f" Data not found in Drive!\n"
        f"Expected: {drive_dataset_path}\n"
        f"Please run Notebook 0 first to prepare data."
    )

# Copy to local for faster access
if local_dataset_path.exists():
    shutil.rmtree(local_dataset_path)

print("   Copying from Drive to local (faster for training)...")
shutil.copytree(drive_dataset_path, local_dataset_path)

# Copy config
shutil.copy2(
    DRIVE_DATA_DIR / "config.json",
    LOCAL_DATA_DIR / "config.json"
)

# Load dataset
dataset = load_from_disk(str(local_dataset_path))
train_dataset = dataset['train']
test_dataset = dataset['test']

# Load config
with open(LOCAL_DATA_DIR / "config.json") as f:
    config = json.load(f)

print(f" Data loaded from Drive!")
print(f"   Train: {len(train_dataset)} samples")
print(f"   Test: {len(test_dataset)} samples")
print(f"   Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Initialize Tokenizer with Optimized MAX_LENGTH
print("\n Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(config['policy_model'])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'  # Important for generation

# OPTIMIZED: Use 224 instead of 512
MAX_LENGTH = 224  # Covers 96.7% of data, 5.2x faster

print(f" Tokenizer loaded: {config['policy_model']}")
print(f"   Vocab size: {len(tokenizer)}")
print(f"   MAX_LENGTH: {MAX_LENGTH} ")


# Tokenize Dataset
def tokenize_function(examples):
    """Tokenize prompts and responses with optimized max_length"""
    prompts = examples['prompt']
    chosen = examples['chosen']
    rejected = examples['rejected']
    
    # Tokenize with optimized MAX_LENGTH
    prompt_tokens = tokenizer(prompts, truncation=True, max_length=MAX_LENGTH)
    chosen_tokens = tokenizer(chosen, truncation=True, max_length=MAX_LENGTH)
    rejected_tokens = tokenizer(rejected, truncation=True, max_length=MAX_LENGTH)
    
    return {
        'input_ids': prompt_tokens['input_ids'],
        'attention_mask': prompt_tokens['attention_mask'],
        'chosen_input_ids': chosen_tokens['input_ids'],
        'rejected_input_ids': rejected_tokens['input_ids'],
    }

print(" Tokenizing dataset...")
tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train"
)
tokenized_test = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=test_dataset.column_names,
    desc="Tokenizing test"
)

print(" Tokenization complete")

# Helper Functions
def get_lora_model(model_name, device='cuda'):
    """Load model with LoRA"""
    print(f"   Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto'
    )
    
    # LoRA config
    lora_config = LoraConfig(
        r=config.get('lora_r', 8),
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def save_model_and_results(model, tokenizer, save_name, metrics, training_time):
    """Save model to both local and Drive"""
    # Save to local first (faster)
    local_path = LOCAL_MODELS_DIR / save_name
    local_path.mkdir(exist_ok=True)
    
    model.save_pretrained(local_path)
    tokenizer.save_pretrained(local_path)
    
    # Save metrics
    results = {
        'metrics': metrics,
        'training_time': training_time,
        'config': config,
        'max_length': MAX_LENGTH  # Document optimization
    }
    
    with open(local_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"    Saved to local: {local_path}")
    
    # Copy to Drive (persistent)
    drive_path = DRIVE_MODELS_DIR / save_name
    if drive_path.exists():
        shutil.rmtree(drive_path)
    
    shutil.copytree(local_path, drive_path)
    print(f"    Copied to Drive: {drive_path}")

print(" Helper functions loaded")

# Baseline SFT (Supervised Fine-Tuning)
print("\n" + "="*60)
print(" STEP 1: Baseline SFT (Supervised Fine-Tuning)")
print("="*60)

# Prepare data for SFT (only use chosen responses)
def prepare_sft_data(examples):
    """Format data for supervised fine-tuning"""
    texts = []
    for prompt, chosen in zip(examples['prompt'], examples['chosen']):
        text = f"Human: {prompt}\n\nAssistant: {chosen}"
        texts.append(text)
    return tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding='max_length')

print(" Preparing SFT data...")
sft_train = train_dataset.map(
    prepare_sft_data,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Preparing SFT data"
)

# Load model
print(" Loading model...")
sft_model = get_lora_model(config['policy_model'])

# Training arguments - OPTIMIZED
sft_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR / "sft"),
    num_train_epochs=config.get('num_epochs', 2),
    per_device_train_batch_size=8,  # Increased from 4 (freed memory)
    gradient_accumulation_steps=2,  # Decreased from 4 (larger batch)
    learning_rate=config.get('learning_rate', 5e-5),
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    remove_unused_columns=False,
    report_to="none",
)

# Trainer
sft_trainer = Trainer(
    model=sft_model,
    args=sft_args,
    train_dataset=sft_train,
    tokenizer=tokenizer,
)

# Train
print(" Starting SFT training...")
start_time = time.time()
sft_result = sft_trainer.train()
sft_time = time.time() - start_time

print(f" SFT complete in {sft_time/60:.1f} minutes")

# Save
save_model_and_results(
    sft_model, 
    tokenizer, 
    "sft_baseline",
    sft_result.metrics,
    sft_time
)

# Clean up
del sft_model, sft_trainer
torch.cuda.empty_cache()

# Baseline DPO (No Privacy)
print("\n" + "="*60)
print(" STEP 2: Baseline DPO (No Privacy)")
print("="*60)

# Load models
print(" Loading models...")
dpo_model = get_lora_model(config['policy_model'])
dpo_ref_model = get_lora_model(config['policy_model'])  # Reference model

# DPO config - OPTIMIZED
dpo_config = DPOConfig(
    output_dir=str(CHECKPOINT_DIR / "dpo_baseline"),
    num_train_epochs=config.get('num_epochs', 2),
    per_device_train_batch_size=8,  # Increased from 4
    gradient_accumulation_steps=2,  # Decreased from 4
    learning_rate=config.get('learning_rate', 5e-5),
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    beta=0.1,  # DPO temperature
    remove_unused_columns=False,
    report_to="none",
    max_length=MAX_LENGTH,  # Use optimized length
    max_prompt_length=MAX_LENGTH // 2,  # Half for prompt
)

# Prepare DPO dataset
def prepare_dpo_data(examples):
    """Format data for DPO"""
    return {
        'prompt': examples['prompt'],
        'chosen': examples['chosen'],
        'rejected': examples['rejected']
    }

print(" Preparing DPO data...")
dpo_train = train_dataset.map(
    prepare_dpo_data,
    batched=True,
    desc="Preparing DPO data"
)

# DPO Trainer
dpo_trainer = DPOTrainer(
    model=dpo_model,
    ref_model=dpo_ref_model,
    args=dpo_config,
    train_dataset=dpo_train,
    tokenizer=tokenizer,
)

# Train
print(" Starting DPO training...")
start_time = time.time()
dpo_result = dpo_trainer.train()
dpo_time = time.time() - start_time

print(f" DPO baseline complete in {dpo_time/60:.1f} minutes")

# Save
save_model_and_results(
    dpo_model,
    tokenizer,
    "dpo_baseline",
    dpo_result.metrics,
    dpo_time
)

# Clean up
del dpo_model, dpo_ref_model, dpo_trainer
torch.cuda.empty_cache()

# DP-DPO Training Function
def train_dp_dpo(epsilon, max_grad_norm=1.0):
    """Train DPO with Differential Privacy"""
    print(f"\n{'='*60}")
    print(f" Training DP-DPO with ε={epsilon}")
    print(f"{'='*60}")
    
    # Load models
    print(" Loading models...")
    model = get_lora_model(config['policy_model'])
    ref_model = get_lora_model(config['policy_model'])
    
    # Make model compatible with Opacus
    model = ModuleValidator.fix(model)
    
    # Training config - OPTIMIZED
    training_args = DPOConfig(
        output_dir=str(CHECKPOINT_DIR / f"dp_dpo_eps{epsilon}"),
        num_train_epochs=config.get('num_epochs',2),
        per_device_train_batch_size=8,  # Increased
        gradient_accumulation_steps=2,  # Decreased
        learning_rate=config.get('learning_rate', 5e-5),
        fp16=False,  # DP doesn't work well with fp16
        logging_steps=50,
        save_strategy="epoch",
        beta=0.1,
        remove_unused_columns=False,
        report_to="none",
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_LENGTH // 2,
    )
    
    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dpo_train,
        tokenizer=tokenizer,
    )
    
    # Add Privacy Engine
    print(" Configuring privacy engine...")
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
        module=trainer.model,
        optimizer=trainer.optimizer,
        data_loader=trainer.get_train_dataloader(),
        epochs=config.get('num_epochs', 2),
        target_epsilon=epsilon,
        target_delta=config.get('delta', 1e-5),
        max_grad_norm=max_grad_norm,
    )
    
    print(f"   Privacy engine configured:")
    print(f"   Target ε: {epsilon}")
    print(f"   δ: {config.get('delta', 1e-5)}")
    print(f"   Max grad norm: {max_grad_norm}")
    
    # Train
    print("  Starting DP-DPO training...")
    start_time = time.time()
    result = trainer.train()
    training_time = time.time() - start_time
    
    # Get final privacy spent
    epsilon_spent = privacy_engine.get_epsilon(config.get('delta', 1e-5))
    print(f"   DP-DPO complete in {training_time/60:.1f} minutes")
    print(f"   Final ε spent: {epsilon_spent:.2f}")
    
    # Save
    save_name = f"dp_dpo_eps{epsilon}"
    metrics = {**result.metrics, 'epsilon_spent': epsilon_spent}
    save_model_and_results(model, tokenizer, save_name, metrics, training_time)
    
    # Clean up
    del model, ref_model, trainer, privacy_engine
    torch.cuda.empty_cache()
    
    return metrics, training_time

# Train DP-DPO ε=8 (Moderate Privacy)
metrics_eps8, time_eps8 = train_dp_dpo(epsilon=8.0)

#% Train DP-DPO ε=1 (Strong Privacy)
metrics_eps1, time_eps1 = train_dp_dpo(epsilon=1.0)

# Optional - Train More Epsilon Values
# Uncomment to train additional privacy budgets

# print("\n Training additional privacy budgets...")

# Train ε=16 (Weak Privacy)
# metrics_eps16, time_eps16 = train_dp_dpo(epsilon=16.0)

# Train ε=4 (Moderate-Strong Privacy)
# metrics_eps4, time_eps4 = train_dp_dpo(epsilon=4.0)

# Summary
print("\n" + "="*60)
print(" DPO TRACK COMPLETE!")
print("="*60)

models_trained = [
    "sft_baseline",
    "dpo_baseline",
    "dp_dpo_eps8.0",
    "dp_dpo_eps1.0",
]

print(f"\n Models trained: {len(models_trained)}")
for model_name in models_trained:
    local_path = LOCAL_MODELS_DIR / model_name
    drive_path = DRIVE_MODELS_DIR / model_name
    
    if drive_path.exists():
        print(f"   ✓ {model_name}")
    elif local_path.exists():
        print(f"     {model_name} (local only - check Drive copy)")
    else:
        print(f"    {model_name} (missing)")

print(f"\n Storage locations:")
print(f"   Local (temporary): {LOCAL_MODELS_DIR}")
print(f"   Drive (persistent): {DRIVE_MODELS_DIR}")

print("\n   Training time summary:")
total_time = sft_time + dpo_time + time_eps8 + time_eps1
print(f"   SFT baseline: {sft_time/60:.1f} min")
print(f"   DPO baseline: {dpo_time/60:.1f} min")
print(f"   DP-DPO ε=8: {time_eps8/60:.1f} min")
print(f"   DP-DPO ε=1: {time_eps1/60:.1f} min")
print(f"   Total: {total_time/60:.1f} min ({total_time/3600:.1f} hours)")

print("\n Optimization impact:")
print(f"   MAX_LENGTH: 224 (vs 512 baseline)")
print(f"   Expected speedup: 5.2x")
print(f"   Coverage: 96.7% of data")

print("\n Next: Run Notebook 2 (RLHF Track)")
print("="*60)