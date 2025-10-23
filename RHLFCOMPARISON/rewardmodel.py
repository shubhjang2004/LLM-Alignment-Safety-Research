"""
RLHF Project - Notebook 2: Reward Model Training
Trains DistilBERT as a preference reward model using pairwise comparisons
"""

# Install required packages
# !pip install transformers datasets torch accelerate wandb -q

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_from_disk
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

print("="*60)
print("RLHF Reward Model Training - DistilBERT")
print("="*60)

# Configuration
class RewardModelConfig:
    # Model
    model_name = "distilgpt2"
    max_length = 512
    
    # Training
    batch_size = 8
    num_epochs = 3
    learning_rate = 1e-5
    warmup_steps = 100
    gradient_accumulation_steps = 4
    max_grad_norm = 1.0
    
    # Paths
    data_path = "./rlhf_compact_data"
    output_dir = "./reward_model_distilgpt2"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = RewardModelConfig()

class RewardModel(nn.Module):
    """
    distilgpt2-based reward model for RLHF
    Takes text input and outputs a scalar reward score
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.transformer.config.hidden_size, 1)
        
        # Initialize reward head
        nn.init.normal_(self.reward_head.weight, std=0.01)
        nn.init.zeros_(self.reward_head.bias)
    
    # Change class RewardModel forward method:
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
    
        # For GPT models: use LAST token instead of first
        sequence_lengths = attention_mask.sum(dim=1) - 1
        hidden_state = outputs.last_hidden_state[range(len(sequence_lengths)), sequence_lengths]
    
        reward = self.reward_head(hidden_state)
        return reward.squeeze(-1)

class RewardDataset(torch.utils.data.Dataset):
    """Dataset for pairwise preference training"""
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Combine prompt with responses
        chosen_text = example['prompt'] + " " + example['chosen']
        rejected_text = example['prompt'] + " " + example['rejected']
        
        # Tokenize
        chosen_encoded = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        rejected_encoded = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoded['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoded['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encoded['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoded['attention_mask'].squeeze(0)
        }

def compute_loss(chosen_rewards, rejected_rewards):
    """
    Compute pairwise ranking loss
    We want chosen_reward > rejected_reward
    """
    # Ranking loss: -log(sigmoid(chosen - rejected))
    return -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

def evaluate_model(model, dataloader, device):
    """Evaluate reward model on validation set"""
    model.eval()
    total_loss = 0
    correct_preferences = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            chosen_ids = batch['chosen_input_ids'].to(device)
            chosen_mask = batch['chosen_attention_mask'].to(device)
            rejected_ids = batch['rejected_input_ids'].to(device)
            rejected_mask = batch['rejected_attention_mask'].to(device)
            
            # Get rewards
            chosen_rewards = model(chosen_ids, chosen_mask)
            rejected_rewards = model(rejected_ids, rejected_mask)
            
            # Compute loss
            loss = compute_loss(chosen_rewards, rejected_rewards)
            total_loss += loss.item()
            
            # Compute accuracy (how often chosen > rejected)
            correct_preferences += (chosen_rewards > rejected_rewards).sum().item()
            total_samples += chosen_rewards.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_preferences / total_samples
    
    return avg_loss, accuracy

def train_reward_model(config):
    """Main training function"""
    print(f"\n🚀 Starting reward model training...")
    print(f"Device: {config.device}")
    
    # Load tokenizer and data
    print("\n📥 Loading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = load_from_disk(config.data_path)
    
    # Create datasets
    train_dataset = RewardDataset(dataset['train'], tokenizer, config.max_length)
    val_dataset = RewardDataset(dataset['validation'], tokenizer, config.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    print("\n🤖 Initializing reward model...")
    model = RewardModel(config.model_name).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
    best_val_accuracy = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            chosen_ids = batch['chosen_input_ids'].to(config.device)
            chosen_mask = batch['chosen_attention_mask'].to(config.device)
            rejected_ids = batch['rejected_input_ids'].to(config.device)
            rejected_mask = batch['rejected_attention_mask'].to(config.device)
            
            # Forward pass
            chosen_rewards = model(chosen_ids, chosen_mask)
            rejected_rewards = model(rejected_ids, rejected_mask)
            
            # Compute loss
            loss = compute_loss(chosen_rewards, rejected_rewards)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            
            # Update weights
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item() * config.gradient_accumulation_steps:.4f}'})
        
        # Epoch summary
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        val_loss, val_accuracy = evaluate_model(model, val_loader, config.device)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"  ✅ New best validation accuracy! Saving model...")
            os.makedirs(config.output_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{config.output_dir}/best_model.pt")
            tokenizer.save_pretrained(config.output_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    return model, history

def plot_training_history(history):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy (Preference Ranking)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/training_history.png", dpi=150)
    print(f"\n📊 Training plots saved to {config.output_dir}/training_history.png")
    plt.show()

def test_reward_model(model, tokenizer, config):
    """Test the trained reward model with examples"""
    print("\n" + "="*60)
    print("TESTING REWARD MODEL")
    print("="*60)
    
    model.eval()
    
    # Test examples
    test_cases = [
        {
            'prompt': "What is the capital of France?",
            'response1': "The capital of France is Paris.",
            'response2': "I don't know."
        },
        {
            'prompt': "How do I make scrambled eggs?",
            'response1': "Break eggs in a bowl, whisk them, heat a pan with butter, pour eggs in, and gently stir until cooked.",
            'response2': "Just put eggs in a microwave."
        },
        {
            'prompt': "What is 2+2?",
            'response1': "2+2 equals 4.",
            'response2': "2+2 is approximately 5."
        }
    ]
    
    with torch.no_grad():
        for i, case in enumerate(test_cases):
            print(f"\n--- Test Case {i+1} ---")
            print(f"Prompt: {case['prompt']}")
            
            # Get rewards for both responses
            text1 = case['prompt'] + " " + case['response1']
            text2 = case['prompt'] + " " + case['response2']
            
            encoded1 = tokenizer(text1, return_tensors='pt', max_length=config.max_length, 
                               truncation=True, padding='max_length').to(config.device)
            encoded2 = tokenizer(text2, return_tensors='pt', max_length=config.max_length,
                               truncation=True, padding='max_length').to(config.device)
            
            reward1 = model(encoded1['input_ids'], encoded1['attention_mask']).item()
            reward2 = model(encoded2['input_ids'], encoded2['attention_mask']).item()
            
            print(f"Response 1: {case['response1']}")
            print(f"  Reward: {reward1:.4f}")
            print(f"Response 2: {case['response2']}")
            print(f"  Reward: {reward2:.4f}")
            print(f"  Preference: {'Response 1' if reward1 > reward2 else 'Response 2'} (Δ={abs(reward1-reward2):.4f})")

# Main execution
if __name__ == "__main__":
    # Train model
    model, history = train_reward_model(config)
    
    # Plot results
    plot_training_history(history)
    
    # Load tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Test model
    test_reward_model(model, tokenizer, config)
    
    print("\n✅ Reward model training complete!")
    print(f"Model saved to: {config.output_dir}")
    print("\nNext: Use this reward model in Notebook 3 for PPO training")