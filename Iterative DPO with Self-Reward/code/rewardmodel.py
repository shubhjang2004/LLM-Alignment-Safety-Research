# Reward Model Training - Complete Code
# Run this entire notebook AFTER SFT training

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
class Config:
    sft_model_dir = "./outputs/sft_model"
    reward_model_dir = "./outputs/reward_model"
    dataset_name = "Anthropic/hh-rlhf"
    num_preference_samples = 5000
    max_length = 512
    reward_epochs = 3
    batch_size = 4
    learning_rate = 1e-4

config = Config()
os.makedirs(config.reward_model_dir, exist_ok=True)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.sft_model_dir)
tokenizer.pad_token = tokenizer.eos_token

# Load preference data
print("Loading preference data...")
dataset = load_dataset(config.dataset_name, split="train")
pref_data = dataset.select(range(min(config.num_preference_samples, len(dataset))))

def format_preference(example):
    try:
        prompt = example['chosen'].split('Assistant:')[0].replace('Human:', '').strip()
        chosen = example['chosen'].split('Assistant:')[-1].strip()
        rejected = example['rejected'].split('Assistant:')[-1].strip()
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}
    except:
        return None

preference_data = []
for ex in tqdm(pref_data, desc="Formatting preferences"):
    formatted = format_preference(ex)
    if formatted and formatted['prompt'] and formatted['chosen'] and formatted['rejected']:
        preference_data.append(formatted)

print(f"Preference dataset size: {len(preference_data)}")

# Custom Dataset
class RewardDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        chosen_text = f"Human: {item['prompt']}\n\nAssistant: {item['chosen']}"
        rejected_text = f"Human: {item['prompt']}\n\nAssistant: {item['rejected']}"
        
        chosen_tokens = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_tokens = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0)
        }

train_dataset = RewardDataset(preference_data, tokenizer, config.max_length)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

# Reward Model Architecture
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        last_hidden = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        reward = self.reward_head(last_hidden)
        return reward

# Load base model
print("Loading SFT model...")
base_model = AutoModelForCausalLM.from_pretrained(
    config.sft_model_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Create reward model
print("Creating reward model...")
reward_model = RewardModel(base_model)
reward_model = reward_model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(reward_model.reward_head.parameters(), lr=config.learning_rate)

# Training loop
print("Starting reward model training...")
reward_model.train()

for epoch in range(config.reward_epochs):
    total_loss = 0
    total_acc = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.reward_epochs}")
    
    for batch in pbar:
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            chosen_reward = reward_model(chosen_ids, chosen_mask)
            rejected_reward = reward_model(rejected_ids, rejected_mask)
            
            loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_model.reward_head.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        acc = (chosen_reward > rejected_reward).float().mean().item()
        total_acc += acc
        
        pbar.set_postfix({"loss": loss.item(), "acc": acc})
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

# Save reward model
print("Saving reward model...")
torch.save({
    'reward_head_state_dict': reward_model.reward_head.state_dict(),
    'config': config.__dict__
}, os.path.join(config.reward_model_dir, "reward_model.pt"))

base_model.save_pretrained(config.reward_model_dir)
tokenizer.save_pretrained(config.reward_model_dir)

print(f"✓ Reward model saved to {config.reward_model_dir}")
print("\n✓ Reward Model Training Complete!")