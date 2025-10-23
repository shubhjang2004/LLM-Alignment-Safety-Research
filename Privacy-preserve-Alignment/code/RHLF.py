




import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import time
from tqdm.auto import tqdm

# Paths
BASE_DIR = Path("/")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

# Load config
with open(DATA_DIR / "config.json") as f:
    config = json.load(f)

print(" Setup complete")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

print(" Loading processed data...")

dataset = load_from_disk(str(DATA_DIR / "hh_rlhf_processed"))
train_dataset = dataset['train']
test_dataset = dataset['test']

print(f" Train: {len(train_dataset)} samples")
print(f" Test: {len(test_dataset)} samples")


print(" Loading tokenizers...")

# Policy model tokenizer
policy_tokenizer = AutoTokenizer.from_pretrained(config['policy_model'])
policy_tokenizer.pad_token = policy_tokenizer.eos_token
policy_tokenizer.padding_side = 'left'

# Reward model tokenizer
reward_tokenizer = AutoTokenizer.from_pretrained(config['reward_model'])
if reward_tokenizer.pad_token is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token

print(f" Policy tokenizer: {config['policy_model']}")
print(f" Reward tokenizer: {config['reward_model']}")


def prepare_reward_data(examples):
    """Format data for reward model training"""
    texts_chosen = []
    texts_rejected = []
    
    for prompt, chosen, rejected in zip(
        examples['prompt'], 
        examples['chosen'], 
        examples['rejected']
    ):
        # Combine prompt + response
        text_chosen = f"{prompt} {chosen}"
        text_rejected = f"{prompt} {rejected}"
        
        texts_chosen.append(text_chosen)
        texts_rejected.append(text_rejected)
    
    # Tokenize
    chosen_tokens = reward_tokenizer(
        texts_chosen,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors=None
    )
    
    rejected_tokens = reward_tokenizer(
        texts_rejected,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors=None
    )
    
    return {
        'input_ids_chosen': chosen_tokens['input_ids'],
        'attention_mask_chosen': chosen_tokens['attention_mask'],
        'input_ids_rejected': rejected_tokens['input_ids'],
        'attention_mask_rejected': rejected_tokens['attention_mask'],
    }

print(" Preparing reward model dataset...")
reward_train_dataset = train_dataset.map(
    prepare_reward_data,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Preparing reward data"
)

print(" Reward dataset prepared")

class RewardModelTrainer:
    """Custom trainer for reward model"""
    
    def __init__(self, model, tokenizer, train_dataset, output_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def compute_loss(self, batch):
        """Compute ranking loss"""
        # Forward pass for chosen
        outputs_chosen = self.model(
            input_ids=batch['input_ids_chosen'],
            attention_mask=batch['attention_mask_chosen']
        )
        reward_chosen = outputs_chosen.logits.squeeze(-1)
        
        # Forward pass for rejected
        outputs_rejected = self.model(
            input_ids=batch['input_ids_rejected'],
            attention_mask=batch['attention_mask_rejected']
        )
        reward_rejected = outputs_rejected.logits.squeeze(-1)
        
        # Ranking loss: chosen should have higher reward
        loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
        
        return loss, reward_chosen.mean().item(), reward_rejected.mean().item()
    
    def train(self, num_epochs=3, batch_size=4, lr=5e-5):
        """Training loop"""
        from torch.utils.data import DataLoader
        
        # Prepare dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Training
        self.model.train()
        device = next(self.model.parameters()).device
        
        total_steps = len(dataloader) * num_epochs
        progress_bar = tqdm(total=total_steps, desc="Training Reward Model")
        
        losses = []
        rewards_chosen_all = []
        rewards_rejected_all = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward + backward
                loss, reward_chosen, reward_rejected = self.compute_loss(batch)
                loss.backward()
                
                # Update
                optimizer.step()
                optimizer.zero_grad()
                
                # Log
                epoch_loss += loss.item()
                losses.append(loss.item())
                rewards_chosen_all.append(reward_chosen)
                rewards_rejected_all.append(reward_rejected)
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'r_chosen': f'{reward_chosen:.3f}',
                    'r_rejected': f'{reward_rejected:.3f}'
                })
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"\nEpoch {epoch+1}/{num_epochs} - Avg Loss: {avg_epoch_loss:.4f}")
        
        progress_bar.close()
        
        # Save metrics
        metrics = {
            'final_loss': losses[-1],
            'avg_reward_chosen': np.mean(rewards_chosen_all),
            'avg_reward_rejected': np.mean(rewards_rejected_all),
            'reward_margin': np.mean(rewards_chosen_all) - np.mean(rewards_rejected_all)
        }
        
        return metrics


print("\n" + "="*60)
print(" STEP 1: Train Reward Model")
print("="*60)

# Load reward model (DistilBERT with regression head)
print("Loading DistilBERT for reward modeling...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    config['reward_model'],
    num_labels=1,  # Regression task
    torch_dtype=torch.float16,
    device_map='auto'
)

print(f" Model loaded: {config['reward_model']}")

# Create trainer
rm_trainer = RewardModelTrainer(
    model=reward_model,
    tokenizer=reward_tokenizer,
    train_dataset=reward_train_dataset,
    output_dir=CHECKPOINT_DIR / "reward_model"
)

# Train
print(" Starting reward model training...")
start_time = time.time()
rm_metrics = rm_trainer.train(
    num_epochs=config['num_epochs'],
    batch_size=config['batch_size'],
    lr=config['learning_rate']
)
rm_time = time.time() - start_time

print(f"\n Reward model training complete in {rm_time/60:.1f} minutes")
print(f"   Reward margin (chosen - rejected): {rm_metrics['reward_margin']:.3f}")

# Save reward model
save_path = MODEL_DIR / "reward_model"
save_path.mkdir(exist_ok=True)
reward_model.save_pretrained(save_path)
reward_tokenizer.save_pretrained(save_path)

# Save metrics
with open(save_path / 'metrics.json', 'w') as f:
    json.dump({**rm_metrics, 'training_time': rm_time}, f, indent=2)

print(f" Reward model saved to {save_path}")

# Clean up
del reward_model, rm_trainer
torch.cuda.empty_cache()


def load_reward_model():
    """Load trained reward model"""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR / "reward_model",
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "reward_model")
    return model, tokenizer

def get_reward_score(prompt, response, reward_model, reward_tokenizer):
    """Get reward score for a prompt-response pair"""
    text = f"{prompt} {response}"
    inputs = reward_tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=512
    ).to(reward_model.device)
    
    with torch.no_grad():
        reward = reward_model(**inputs).logits.squeeze(-1)
    
    return reward.item()

def prepare_rlhf_prompts(dataset):
    """Extract just prompts for RLHF"""
    return [ex['prompt'] for ex in dataset]

rlhf_train_prompts = prepare_rlhf_prompts(train_dataset)
print(f" Prepared {len(rlhf_train_prompts)} prompts for RLHF")

# PPO Training Function (Baseline - No Privacy)
def train_ppo_baseline(train_prompts, num_epochs=3):
    """Train PPO without differential privacy"""
    print(f"\n{'='*60}")
    print(f" Training Baseline RLHF (PPO, No Privacy)")
    print(f"{'='*60}")
    
    # Load models
    print("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        config['policy_model'],
        torch_dtype=torch.float16,
        device_map='auto'
    )
    
    # Add value head for PPO
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(policy_model)
    
    # LoRA
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    policy_model.pretrained_model = get_peft_model(
        policy_model.pretrained_model, 
        lora_config
    )
    
    # Load reward model
    print("Loading reward model...")
    reward_model, reward_tok = load_reward_model()
    reward_model.eval()
    
    # PPO config
    ppo_config = PPOConfig(
        model_name=config['policy_model'],
        learning_rate=1e-5,
        batch_size=config['batch_size'],
        mini_batch_size=config['batch_size'],
        gradient_accumulation_steps=4,
        ppo_epochs=4,
        max_grad_norm=0.5,
        remove_unused_columns=False,
    )
    
    # PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        tokenizer=policy_tokenizer,
    )
    
    # Generation config
    generation_kwargs = {
        'max_new_tokens': 128,
        'do_sample': True,
        'top_p': 0.9,
        'temperature': 0.7,
        'pad_token_id': policy_tokenizer.pad_token_id,
    }
    
    # Training loop
    print(" Starting PPO training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs}")
        
        # Process in batches
        batch_size = ppo_config.batch_size
        for i in tqdm(range(0, len(train_prompts), batch_size), desc=f"Epoch {epoch+1}"):
            batch_prompts = train_prompts[i:i+batch_size]
            
            # Tokenize prompts
            prompt_tensors = [
                policy_tokenizer.encode(prompt, return_tensors='pt').squeeze(0)
                for prompt in batch_prompts
            ]
            
            # Generate responses
            response_tensors = []
            for prompt_tensor in prompt_tensors:
                prompt_tensor = prompt_tensor.to(policy_model.pretrained_model.device)
                response = policy_model.generate(
                    prompt_tensor.unsqueeze(0),
                    **generation_kwargs
                )
                response_tensors.append(response.squeeze())
            
            # Decode responses
            responses = [
                policy_tokenizer.decode(r, skip_special_tokens=True)
                for r in response_tensors
            ]
            
            # Get rewards
            rewards = []
            for prompt, response in zip(batch_prompts, responses):
                reward = get_reward_score(prompt, response, reward_model, reward_tok)
                rewards.append(torch.tensor(reward))
            
            # PPO step
            stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards)
            
            # Log every 50 steps
            if i % 50 == 0:
                avg_reward = torch.stack(rewards).mean().item()
                print(f"Step {i}: Avg Reward = {avg_reward:.3f}")
    
    training_time = time.time() - start_time
    print(f"\n PPO training complete in {training_time/60:.1f} minutes")
    
    # Save model
    save_path = MODEL_DIR / "rlhf_baseline"
    save_path.mkdir(exist_ok=True)
    policy_model.save_pretrained(save_path)
    policy_tokenizer.save_pretrained(save_path)
    
    metrics = {
        'training_time': training_time,
        'num_epochs': num_epochs
    }
    
    with open(save_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f" Model saved to {save_path}")
    
    # Clean up
    del policy_model, reward_model, ppo_trainer
    torch.cuda.empty_cache()
    
    return metrics, training_time


metrics_rlhf, time_rlhf = train_ppo_baseline(
    rlhf_train_prompts,
    num_epochs=config['num_epochs']
)


def train_dp_ppo(train_prompts, epsilon, num_epochs=3):
    """Train PPO with Differential Privacy"""
    print(f"\n{'='*60}")
    print(f" Training DP-RLHF with ε={epsilon}")
    print(f"{'='*60}")
    
    # Load models
    print("Loading policy model...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        config['policy_model'],
        device_map='auto'
    )
    
    # Make compatible with Opacus
    policy_model = ModuleValidator.fix(policy_model)
    
    # LoRA
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    policy_model = get_peft_model(policy_model, lora_config)
    
    # Load reward model
    print("Loading reward model...")
    reward_model, reward_tok = load_reward_model()
    reward_model.eval()
    
    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5)
    
    # Privacy Engine
    privacy_engine = PrivacyEngine()
    
    # Note: For full DP-PPO, we need custom implementation
    # This is a simplified version showing the privacy mechanics
    print("  Note: This is simplified DP-RLHF")
    print("    Full DP-PPO requires custom implementation of PPO loop with DP")
    
    # For this demo, we'll do DP-supervised learning on preferred trajectories
    # Real DP-RLHF is more complex and would need custom PPO implementation
    
    print(" Starting DP-RLHF training (simplified)...")
    start_time = time.time()
    
    # Simplified training: Generate trajectories and do supervised learning with DP
    policy_model.train()
    device = next(policy_model.parameters()).device
    
    # Make dataloader (for DP per-sample gradients)
    from torch.utils.data import DataLoader, TensorDataset
    
    # Sample some trajectories with current policy
    all_texts = []
    print("Generating training trajectories...")
    for prompt in tqdm(train_prompts[:1000], desc="Generating"):  # Sample subset
        inputs = policy_tokenizer(prompt, return_tensors='pt').to(device)
        outputs = policy_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7
        )
        text = policy_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Get reward
        reward = get_reward_score(prompt, text, reward_model, reward_tok)
        
        # Only keep high-reward trajectories
        if reward > 0.5:
            all_texts.append(text)
    
    print(f"Generated {len(all_texts)} high-reward trajectories")
    
    # Tokenize for supervised learning
    tokenized = policy_tokenizer(
        all_texts,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )
    
    dataset = TensorDataset(
        tokenized['input_ids'],
        tokenized['attention_mask']
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Attach privacy engine
    policy_model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
        module=policy_model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=num_epochs,
        target_epsilon=epsilon,
        target_delta=config.get('delta', 1e-5),
        max_grad_norm=1.0,
    )
    
    print(f" Privacy engine configured for ε={epsilon}")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids, attention_mask = batch
            
            # Forward pass
            outputs = policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    epsilon_spent = privacy_engine.get_epsilon(config.get('delta', 1e-5))
    
    print(f"\n DP-RLHF complete in {training_time/60:.1f} minutes")
    print(f"   Final ε spent: {epsilon_spent:.2f}")
    
    # Save model
    save_path = MODEL_DIR / f"dp_rlhf_eps{epsilon}"
    save_path.mkdir(exist_ok=True)
    policy_model.save_pretrained(save_path)
    policy_tokenizer.save_pretrained(save_path)
    
    metrics = {
        'training_time': training_time,
        'epsilon_target': epsilon,
        'epsilon_spent': epsilon_spent,
        'num_epochs': num_epochs
    }
    
    with open(save_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f" Model saved to {save_path}")
    
    # Clean up
    del policy_model, reward_model, privacy_engine
    torch.cuda.empty_cache()
    
    return metrics, training_time

metrics_dp_rlhf_8, time_dp_rlhf_8 = train_dp_ppo(
    rlhf_train_prompts,
    epsilon=8.0,
    num_epochs=config['num_epochs']
)


metrics_dp_rlhf_1, time_dp_rlhf_1 = train_dp_ppo(
    rlhf_train_prompts,
    epsilon=1.0,
    num_epochs=config['num_epochs']
)


print("\n" + "="*60)
print(" RLHF TRACK COMPLETE!")
print("="*60)

models_trained = [
    "reward_model",
    "rlhf_baseline",
    "dp_rlhf_eps8.0",
    "dp_rlhf_eps1.0",
]

print(f"\n Models trained: {len(models_trained)}")
for model_name in models_trained:
    model_path = MODEL_DIR / model_name
    if model_path.exists():
        print(f"   ✓ {model_name}")

print(f"\n All models saved to: {MODEL_DIR}")
print("\n Next: Run Notebook 3 (Privacy Attacks)")
print("="*60)