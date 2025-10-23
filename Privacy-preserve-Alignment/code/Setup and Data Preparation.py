

"""
Project 4: Privacy-Preserving Alignment
Notebook 0: Setup and Data Preparation

Purpose: Install dependencies, load data, create splits
Time: ~5 minutes
"""



import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Configuration
class Config:
    """Global configuration"""
    # Paths
    BASE_DIR = Path("/content")
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # Model configs
    POLICY_MODEL = "gpt2"  # GPT-2 Small (124M)
    REWARD_MODEL = "distilbert-base-uncased"  # DistilBERT (66M)
    
    # Data configs
    DATASET_NAME = "Anthropic/hh-rlhf"
    NUM_SAMPLES = 20000
    TRAIN_SPLIT = 0.9
    SEED = 42
    
    # Training configs
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3
    
    # LoRA configs
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["c_attn", "c_proj"]  # For GPT-2
    
    # Privacy configs
    EPSILON_VALUES = [1.0, 4.0, 8.0, 16.0]
    DELTA = 1e-5  # For DP
    MAX_GRAD_NORM = 1.0  # Gradient clipping
    
    # Evaluation configs
    EVAL_BATCH_SIZE = 8
    NUM_EVAL_SAMPLES = 500

# Save config
config = Config()
print(" Configuration initialized")
print(f"Data directory: {config.DATA_DIR}")
print(f"Models will be saved to: {config.MODEL_DIR}")

# Load Dataset
print(" Loading HH-RLHF dataset...")

# Load dataset
dataset = load_dataset(config.DATASET_NAME, split="train")
print(f"Total samples in dataset: {len(dataset)}")

# Sample subset
if len(dataset) > config.NUM_SAMPLES:
    dataset = dataset.shuffle(seed=config.SEED).select(range(config.NUM_SAMPLES))
    print(f"Sampled {config.NUM_SAMPLES} examples")

# Show example
print("\n Example from dataset:")
example = dataset[0]
print(f"Chosen: {example['chosen'][:200]}...")
print(f"Rejected: {example['rejected'][:200]}...")

# Process and Split Data
from datasets import Dataset

def process_sample(example):
    """Extract prompt and responses from conversational format"""
    # HH-RLHF format: "Human: ... Assistant: ..."
    # Extract the prompt (Human's question)
    chosen_text = example['chosen']
    rejected_text = example['rejected']
    
    # Find where assistant response starts
    if "Assistant:" in chosen_text:
        parts = chosen_text.split("Assistant:")
        prompt = parts[0].replace("Human:", "").strip()
        chosen_response = parts[1].strip() if len(parts) > 1 else ""
    else:
        prompt = chosen_text[:100]  # Fallback
        chosen_response = chosen_text
    
    if "Assistant:" in rejected_text:
        rejected_response = rejected_text.split("Assistant:")[1].strip()
    else:
        rejected_response = rejected_text
    
    return {
        'prompt': prompt,
        'chosen': chosen_response,
        'rejected': rejected_response
    }

# Process dataset
print("\n Processing dataset...")
processed_data = [process_sample(ex) for ex in dataset]

# Convert to Dataset
processed_dataset = Dataset.from_list(processed_data)

# Train/test split
split_idx = int(len(processed_dataset) * config.TRAIN_SPLIT)
train_dataset = processed_dataset.select(range(split_idx))
test_dataset = processed_dataset.select(range(split_idx, len(processed_dataset)))

print(f" Train samples: {len(train_dataset)}")
print(f" Test samples: {len(test_dataset)}")

# Show processed example
print("\n Processed example:")
print(f"Prompt: {train_dataset[0]['prompt'][:150]}...")
print(f"Chosen: {train_dataset[0]['chosen'][:150]}...")
print(f"Rejected: {train_dataset[0]['rejected'][:150]}...")

# Save Data
print("\n Saving processed data...")

# Save as JSON
train_data = [dict(ex) for ex in train_dataset]
test_data = [dict(ex) for ex in test_dataset]

with open(config.DATA_DIR / "train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open(config.DATA_DIR / "test.json", "w") as f:
    json.dump(test_data, f, indent=2)

# Also save as Hugging Face dataset
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})
dataset_dict.save_to_disk(str(config.DATA_DIR / "hh_rlhf_processed"))

print("   Data saved successfully!")
print(f"   - Train: {config.DATA_DIR / 'train.json'}")
print(f"   - Test: {config.DATA_DIR / 'test.json'}")
print(f"   - HF Dataset: {config.DATA_DIR / 'hh_rlhf_processed'}")

# Data Statistics
print("\n Dataset Statistics:")

# Length distributions
train_lengths = [len(ex['prompt'].split()) + len(ex['chosen'].split()) 
                 for ex in train_dataset]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(train_lengths, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Total tokens (approx)')
plt.ylabel('Frequency')
plt.title('Distribution of Sample Lengths')
plt.axvline(np.mean(train_lengths), color='r', linestyle='--', 
            label=f'Mean: {np.mean(train_lengths):.0f}')
plt.legend()

plt.subplot(1, 2, 2)
prompt_lengths = [len(ex['prompt'].split()) for ex in train_dataset]
response_lengths = [len(ex['chosen'].split()) for ex in train_dataset]
plt.boxplot([prompt_lengths, response_lengths], labels=['Prompts', 'Responses'])
plt.ylabel('Length (words)')
plt.title('Prompt vs Response Lengths')

plt.tight_layout()
plt.savefig(config.RESULTS_DIR / 'data_statistics.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Mean length: {np.mean(train_lengths):.1f} words")
print(f"Max length: {max(train_lengths)} words")
print(f"Min length: {min(train_lengths)} words")

print("\n Saving configuration...")

config_dict = {
    'policy_model': config.POLICY_MODEL,
    'reward_model': config.REWARD_MODEL,
    'num_samples': config.NUM_SAMPLES,
    'train_samples': len(train_dataset),
    'test_samples': len(test_dataset),
    'max_length': config.MAX_LENGTH,
    'batch_size': config.BATCH_SIZE,
    'learning_rate': config.LEARNING_RATE,
    'num_epochs': config.NUM_EPOCHS,
    'lora_r': config.LORA_R,
    'epsilon_values': config.EPSILON_VALUES,
    'seed': config.SEED,
}

with open(config.DATA_DIR / "config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

print(" Configuration saved!")


print("\n" + "="*60)
print(" SETUP COMPLETE!")
print("="*60)
print(f" Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
print(f" Models: {config.POLICY_MODEL} (policy), {config.REWARD_MODEL} (reward)")
print(f" Privacy budgets to test: {config.EPSILON_VALUES}")
print(f" All data saved to: {config.DATA_DIR}")
print("\n Ready to start training!")
print("   Next: Run Notebook 1 (DPO Track)")
print("="*60)