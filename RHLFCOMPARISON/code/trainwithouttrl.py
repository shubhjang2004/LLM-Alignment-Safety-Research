"""



"""

# ============================================================================
# CELL 1: Setup and Installation
# ============================================================================


# ============================================================================
# CELL 2: Imports
# ============================================================================

import os
import json
import gc
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)
from datasets import load_from_disk

print("="*70)
print(" PPO Training for RLHF")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# CELL 3: Configuration
# ============================================================================

@dataclass
class PPOConfig:
    """PPO Training Configuration"""
    
    # Paths
    data_path: str = "./rlhf_compact_data"
    reward_model_path: str = "./reward_model_distilgpt2"
    reward_model_name: str = "distilgpt2"
    output_dir: str = "./ppo_trained"
    
    # Model
    model_name: str = "gpt2"  # Can be: gpt2, gpt2-medium, gpt2-large
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    ppo_epochs: int = 4
    num_training_steps: int = 200
    max_grad_norm: float = 1.0
    
    # PPO specific
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip parameter
    value_loss_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    kl_target: float = 0.01  # Target KL divergence
    kl_coef: float = 0.1  # KL penalty coefficient
    
    # Generation
    max_new_tokens: int = 40
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    # Data
    max_prompt_length: int = 256
    max_response_length: int = 512
    num_prompts: int = 1000
    
    # Checkpointing
    save_frequency: int = 50
    eval_frequency: int = 20
    
    # Memory management
    gradient_accumulation_steps: int = 1
    fp16: bool = True  # Use mixed precision
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

# Initialize config
config = PPOConfig()

print("\n Configuration:")
print(f"  Model: {config.model_name}")
print(f"  Batch size: {config.batch_size}")
print(f"  Training steps: {config.num_training_steps}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Output: {config.output_dir}")

# ============================================================================
# CELL 4: Reward Model
# ============================================================================

class RewardModel(nn.Module):
    """
    Reward model that takes text and outputs a scalar reward.
    Uses a pretrained transformer + linear head.
    """
    def __init__(self, model_name_or_path: str):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.transformer.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1)
        
        # Initialize head
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            rewards: [batch_size]
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        
        # Get last non-padded token's hidden state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        
        last_hidden_states = outputs.last_hidden_state[
            torch.arange(batch_size, device=input_ids.device),
            sequence_lengths
        ]
        
        rewards = self.reward_head(last_hidden_states).squeeze(-1)
        return rewards

def load_reward_model(config: PPOConfig) -> RewardModel:
    """Load the trained reward model"""
    print(f"\n Loading reward model from: {config.reward_model_path}")
    
    model = RewardModel(config.reward_model_name)
    
    # Try to load trained weights
    weights_path = os.path.join(config.reward_model_path, "best_model.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=config.device)
        model.load_state_dict(state_dict)
        print("  Loaded trained reward model weights")
    else:
        print("  Warning: No trained weights found, using random initialization")
        print(f"   Looking for: {weights_path}")
    
    model.to(config.device)
    model.eval()
    
    # Freeze reward model
    for param in model.parameters():
        param.requires_grad = False
    
    return model

# Load reward model
reward_model = load_reward_model(config)
print(f"  Reward model loaded with {sum(p.numel() for p in reward_model.parameters()):,} parameters")

# ============================================================================
# CELL 5: Value Network
# ============================================================================

class ValueHead(nn.Module):
    """
    Value head for estimating state values.
    This is separate from the policy for stability.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            values: [batch_size, seq_len]
        """
        values = self.value_head(hidden_states).squeeze(-1)
        return values

# ============================================================================
# CELL 6: Policy Model with Value Head
# ============================================================================

class PolicyModelWithValue(nn.Module):
    """
    Combined policy and value model.
    - Policy: The causal LM that generates text
    - Value: Estimates the value of states
    """
    def __init__(self, model_name: str, config: PPOConfig):
        super().__init__()
        
        # Load pretrained causal LM
        dtype = torch.float16 if config.fp16 and torch.cuda.is_available() else torch.float32
        self.policy = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        )
        
        # Add value head
        hidden_size = self.policy.config.n_embd
        self.value_head = ValueHead(hidden_size)
        
        self.config_model = self.policy.config
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_values: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through policy and optionally value head.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_values: Whether to compute values
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            values: [batch_size, seq_len] if return_values else None
        """
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_values,
            use_cache=False  # Important: disable KV cache for training
        )
        
        logits = outputs.logits
        
        if return_values:
            hidden_states = outputs.hidden_states[-1]  # Last layer
            values = self.value_head(hidden_states)
            return logits, values
        
        return logits, None
    
    def generate(self, **kwargs):
        """Wrapper for generation"""
        return self.policy.generate(**kwargs)

# Load policy
print("\n Loading policy model...")
policy_model = PolicyModelWithValue(config.model_name, config)
policy_model.to(config.device)

print(f"  Policy loaded: {config.model_name}")
print(f"   Total parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
print(f"   Trainable parameters: {sum(p.numel() for p in policy_model.parameters() if p.requires_grad):,}")

# ============================================================================
# CELL 7: Reference Model (for KL penalty)
# ============================================================================

print("\n Loading reference model (for KL divergence)...")

# Reference model is a frozen copy of initial policy
reference_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.float16 if config.fp16 and torch.cuda.is_available() else torch.float32
)
reference_model.to(config.device)
reference_model.eval()

# Freeze all parameters
for param in reference_model.parameters():
    param.requires_grad = False

print(f"  Reference model loaded and frozen")

# ============================================================================
# CELL 8: Load Dataset
# ============================================================================

print("\n" + "="*70)
print(" Loading Dataset")
print("="*70)

# Load dataset
print(f"\n Loading from: {config.data_path}")
dataset = load_from_disk(config.data_path)

# Extract prompts
prompts = []
for i, example in enumerate(dataset['train']):
    if len(prompts) >= config.num_prompts:
        break
    if 'prompt' in example:
        prompts.append(example['prompt'])
    elif 'text' in example:
        prompts.append(example['text'])
    else:
        # Use first text field
        prompts.append(list(example.values())[0])

print(f"  Loaded {len(prompts)} prompts")

# Sample some prompts to show
print("\n Sample prompts:")
for i in range(min(3, len(prompts))):
    print(f"  {i+1}. {prompts[i][:100]}...")

# ============================================================================
# CELL 9: Tokenizers
# ============================================================================

print("\n Loading tokenizers...")

# Policy tokenizer
policy_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
if policy_tokenizer.pad_token is None:
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
policy_tokenizer.padding_side = "left"  # Left padding for generation

# Reward tokenizer
reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
if reward_tokenizer.pad_token is None:
    reward_tokenizer.pad_token = reward_tokenizer.eos_token

print(f"  Policy tokenizer: {config.model_name}")
print(f"  Reward tokenizer: {config.reward_model_name}")

# ============================================================================
# CELL 10: PPO Helper Functions
# ============================================================================

def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: [batch_size, seq_len]
        values: [batch_size, seq_len]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages: [batch_size, seq_len]
        returns: [batch_size, seq_len]
    """
    batch_size, seq_len = rewards.shape
    
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    # Compute advantages using GAE
    gae = 0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * gae_lambda * gae
        advantages[:, t] = gae
        returns[:, t] = advantages[:, t] + values[:, t]
    
    return advantages, returns

def compute_kl_divergence(
    logprobs_new: torch.Tensor,
    logprobs_old: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between old and new policies.
    KL(old || new) = sum(old * (log(old) - log(new)))
    
    But we use a simpler approximation:
    KL ≈ mean(logprobs_old - logprobs_new)
    """
    return (logprobs_old - logprobs_new).mean()

def compute_policy_loss(
    logprobs_new: torch.Tensor,
    logprobs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float
) -> torch.Tensor:
    """
    Compute clipped PPO policy loss.
    
    Args:
        logprobs_new: Log probs from current policy
        logprobs_old: Log probs from old policy
        advantages: Advantage estimates
        clip_epsilon: Clipping parameter
        
    Returns:
        loss: Scalar loss
    """
    # Compute probability ratio
    ratio = torch.exp(logprobs_new - logprobs_old)
    
    # Clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    
    # Take minimum (pessimistic bound)
    policy_loss = -torch.min(surr1, surr2).mean()
    
    return policy_loss

def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor
) -> torch.Tensor:
    """Compute value function loss (MSE)"""
    return F.mse_loss(values, returns)

def compute_entropy_bonus(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy bonus to encourage exploration.
    Higher entropy = more random = more exploration
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy

# ============================================================================
# CELL 11: Generate and Score Function
# ============================================================================

def generate_responses(
    prompts: List[str],
    policy_model: PolicyModelWithValue,
    tokenizer: AutoTokenizer,
    config: PPOConfig
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Generate responses for a batch of prompts.
    
    Returns:
        prompt_ids: Tokenized prompts
        response_ids: Generated response tokens
        response_texts: Decoded response texts
    """
    # Tokenize prompts
    prompt_encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_prompt_length
    ).to(config.device)
    
    # Generate
    policy_model.eval()
    with torch.no_grad():
        generated_ids = policy_model.generate(
            input_ids=prompt_encodings['input_ids'],
            attention_mask=prompt_encodings['attention_mask'],
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Extract only generated part
    prompt_length = prompt_encodings['input_ids'].shape[1]
    response_ids = generated_ids[:, prompt_length:]
    
    # Decode
    response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    return prompt_encodings['input_ids'], response_ids, response_texts

def compute_rewards(
    prompts: List[str],
    responses: List[str],
    reward_model: RewardModel,
    reward_tokenizer: AutoTokenizer,
    device: str
) -> torch.Tensor:
    """
    Compute rewards for prompt-response pairs.
    
    Returns:
        rewards: [batch_size] tensor of rewards
    """
    # Combine prompts and responses
    full_texts = [p + " " + r for p, r in zip(prompts, responses)]
    
    # Tokenize
    encodings = reward_tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_response_length
    ).to(device)
    
    # Get rewards
    with torch.no_grad():
        rewards = reward_model(
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask']
        )
    
    return rewards

# ============================================================================
# CELL 12: PPO Training Step
# ============================================================================

def ppo_step(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    rewards: torch.Tensor,
    policy_model: PolicyModelWithValue,
    reference_model: AutoModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig
) -> Dict[str, float]:
    """
    Perform one PPO training step.
    
    This implements the core PPO algorithm:
    1. Compute old log probs and values
    2. For K epochs:
        a. Compute new log probs and values
        b. Compute advantages using GAE
        c. Compute policy loss (clipped objective)
        d. Compute value loss
        e. Compute KL penalty
        f. Backprop and update
    
    Returns:
        Dictionary of training statistics
    """
    batch_size = prompt_ids.shape[0]
    
    # Concatenate prompt and response
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(full_ids)
    
    # === Get old policy log probs (frozen) ===
    policy_model.eval()
    with torch.no_grad():
        # Old policy logits
        old_logits, old_values = policy_model(
            full_ids,
            attention_mask,
            return_values=True
        )
        
        # Old log probs for generated tokens
        old_logprobs = F.log_softmax(old_logits, dim=-1)
        # Get log prob of actual tokens
        old_selected_logprobs = old_logprobs.gather(
            dim=-1,
            index=full_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Reference model log probs (for KL)
        ref_logits = reference_model(
            full_ids,
            attention_mask
        ).logits
        ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        ref_selected_logprobs = ref_logprobs.gather(
            dim=-1,
            index=full_ids.unsqueeze(-1)
        ).squeeze(-1)
    
    # === Prepare rewards ===
    # Rewards are only non-zero at the end of generation
    reward_tensor = torch.zeros_like(full_ids, dtype=torch.float)
    reward_tensor[:, -1] = rewards  # Reward at final token
    
    # === PPO epochs ===
    stats = {
        'policy_loss': 0.0,
        'value_loss': 0.0,
        'entropy': 0.0,
        'kl_div': 0.0,
        'total_loss': 0.0
    }
    
    policy_model.train()
    
    for epoch in range(config.ppo_epochs):
        # Forward pass with current policy
        new_logits, new_values = policy_model(
            full_ids,
            attention_mask,
            return_values=True
        )
        
        # New log probs
        new_logprobs = F.log_softmax(new_logits, dim=-1)
        new_selected_logprobs = new_logprobs.gather(
            dim=-1,
            index=full_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute advantages
        advantages, returns = compute_advantages(
            reward_tensor,
            old_values.detach(),
            config.gamma,
            config.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss (PPO clipped objective)
        policy_loss = compute_policy_loss(
            new_selected_logprobs,
            old_selected_logprobs.detach(),
            advantages.detach(),
            config.clip_epsilon
        )
        
        # Value loss
        value_loss = compute_value_loss(new_values, returns.detach())
        
        # Entropy bonus (encourage exploration)
        entropy = compute_entropy_bonus(new_logits)
        
        # KL divergence from reference policy
        kl_div = compute_kl_divergence(
            new_selected_logprobs,
            ref_selected_logprobs.detach()
        )
        
        # Total loss
        loss = (
            policy_loss +
            config.value_loss_coef * value_loss -
            config.entropy_coef * entropy +
            config.kl_coef * kl_div
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy_model.parameters(),
            config.max_grad_norm
        )
        optimizer.step()
        
        # Accumulate stats
        stats['policy_loss'] += policy_loss.item()
        stats['value_loss'] += value_loss.item()
        stats['entropy'] += entropy.item()
        stats['kl_div'] += kl_div.item()
        stats['total_loss'] += loss.item()
    
    # Average over epochs
    for key in stats:
        stats[key] /= config.ppo_epochs
    
    return stats

# ============================================================================
# CELL 13: Main Training Loop
# ============================================================================

print("\n" + "="*70)
print(" Starting PPO Training")
print("="*70)

# Setup optimizer
optimizer = torch.optim.Adam(
    policy_model.parameters(),
    lr=config.learning_rate
)

# Training history
history = {
    'rewards': [],
    'policy_loss': [],
    'value_loss': [],
    'entropy': [],
    'kl_div': [],
    'total_loss': []
}

# Progress bar
progress_bar = tqdm(total=config.num_training_steps, desc="PPO Training")

# Training loop
step = 0
prompt_idx = 0

while step < config.num_training_steps:
    # === Sample batch of prompts ===
    batch_prompts = []
    for _ in range(config.batch_size):
        batch_prompts.append(prompts[prompt_idx % len(prompts)])
        prompt_idx += 1
    
    # === Generate responses ===
    prompt_ids, response_ids, response_texts = generate_responses(
        batch_prompts,
        policy_model,
        policy_tokenizer,
        config
    )
    
    # === Compute rewards ===
    rewards = compute_rewards(
        batch_prompts,
        response_texts,
        reward_model,
        reward_tokenizer,
        config.device
    )
    
    # === PPO update ===
    stats = ppo_step(
        prompt_ids,
        response_ids,
        rewards,
        policy_model,
        reference_model,
        optimizer,
        config
    )
    
    # === Log statistics ===
    mean_reward = rewards.mean().item()
    history['rewards'].append(mean_reward)
    history['policy_loss'].append(stats['policy_loss'])
    history['value_loss'].append(stats['value_loss'])
    history['entropy'].append(stats['entropy'])
    history['kl_div'].append(stats['kl_div'])
    history['total_loss'].append(stats['total_loss'])
    
    # Update progress bar
    progress_bar.update(1)
    progress_bar.set_postfix({
        'reward': f'{mean_reward:.3f}',
        'policy_loss': f'{stats["policy_loss"]:.3f}',
        'kl': f'{stats["kl_div"]:.4f}'
    })
    
    # === Print detailed stats ===
    if step % config.eval_frequency == 0:
        print(f"\n Step {step}:")
        print(f"   Reward: {mean_reward:.4f}")
        print(f"   Policy Loss: {stats['policy_loss']:.4f}")
        print(f"   Value Loss: {stats['value_loss']:.4f}")
        print(f"   Entropy: {stats['entropy']:.4f}")
        print(f"   KL Div: {stats['kl_div']:.4f}")
        
        # Show a sample generation
        print(f"   Sample:")
        print(f"     Prompt: {batch_prompts[0][:80]}...")
        print(f"     Response: {response_texts[0][:80]}...")
    
    # === Save checkpoint ===
    if (step + 1) % config.save_frequency == 0:
        checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{step+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        policy_model.policy.save_pretrained(checkpoint_dir)
        policy_tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training history
        with open(os.path.join(checkpoint_dir, "training_history.json"), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n  Checkpoint saved to {checkpoint_dir}")
    
    # === Memory management ===
    if step % 10 == 0:
        torch.cuda.empty_cache()
        gc.collect()
    
    step += 1

progress_bar.close()

print("\n" + "="*70)
print(" Training Complete!")
print("="*70)

# ============================================================================
# CELL 14: Save Final Model
# ============================================================================

print("\n Saving final model...")

final_dir = os.path.join(config.output_dir, "final")
os.makedirs(final_dir, exist_ok=True)

# Save policy
policy_model.policy.save_pretrained(final_dir)
policy_tokenizer.save_pretrained(final_dir)

# Save full history
with open(os.path.join(config.output_dir, "training_history.json"), 'w') as f:
    json.dump(history, f, indent=2)

# Save config
with open(os.path.join(config.output_dir, "config.json"), 'w') as f:
    json.dump({
        'model_name': config.model_name,
        'num_steps': config.num_training_steps,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'final_reward': history['rewards'][-1],
        'avg_reward_last_20': float(np.mean(history['rewards'][-20:]))
    }, f, indent=2)

print(f"  Final model saved to: {final_dir}")
print(f"  Training history saved")

# Print summary
print(f"\n Training Summary:")
print(f"   Total steps: {step}")
print(f"   Final reward: {history['rewards'][-1]:.4f}")
print(f"   Average reward (last 20 steps): {np.mean(history['rewards'][-20:]):.4f}")
print(f"   Final KL divergence: {history['kl_div'][-1]:.4f}")

# ============================================================================
# CELL 15: Plot Training Curves
# ============================================================================

print("\n Generating training plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('PPO Training Curves', fontsize=16)

# Rewards
axes[0, 0].plot(history['rewards'], alpha=0.7)
axes[0, 0].set_title('Rewards')
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].grid(True, alpha=0.3)

# Policy Loss
axes[0, 1].plot(history['policy_loss'], alpha=0.7)
axes[0, 1].set_title('Policy Loss')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)

# Value Loss
axes[0, 2].plot(history['value_loss'], alpha=0.7)
axes[0, 2].set_title('Value Loss')
axes[0, 2].set_xlabel('Step')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].grid(True, alpha=0.3)

# Entropy
axes[1, 0].plot(history['entropy'], alpha=0.7)
axes[1, 0].set_title('Entropy')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Entropy')
axes[1, 0].grid(True, alpha=0.3)

# KL Divergence
axes[1, 1].plot(history['kl_div'], alpha=0.7)
axes[1, 1].axhline(y=config.kl_target, color='r', linestyle='--', label='Target KL')
axes[1, 1].set_title('KL Divergence')
axes[1, 1].set_xlabel('Step')
axes[1, 1].set_ylabel('KL')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Total Loss
axes[1, 2].plot(history['total_loss'], alpha=0.7)
axes[1, 2].set_title('Total Loss')
axes[1, 2].set_xlabel('Step')
axes[1, 2].set_ylabel('Loss')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.output_dir, 'training_curves.png'), dpi=150)
print(f"  Plots saved to: {config.output_dir}/training_curves.png")
plt.show()

# ============================================================================
# CELL 16: Test the Trained Model
# ============================================================================

print("\n" + "="*70)
print(" Testing Trained Model")
print("="*70)

test_prompts = [
    "How can I improve my health?",
    "What's the best way to learn programming?",
    "Tell me about climate change.",
    "How do I start a business?",
    "What are the benefits of exercise?"
]

policy_model.eval()

print("\n Generating responses...\n")

for i, prompt in enumerate(test_prompts, 1):
    print(f"{i}. Prompt: {prompt}")
    
    # Tokenize
    inputs = policy_tokenizer(
        prompt,
        return_tensors="pt"
    ).to(config.device)
    
    # Generate
    with torch.no_grad():
        output_ids = policy_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=80,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=policy_tokenizer.pad_token_id
        )
    
    # Decode
    response = policy_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Compute reward for this response
    response_only = response[len(prompt):].strip()
    reward = compute_rewards(
        [prompt],
        [response_only],
        reward_model,
        reward_tokenizer,
        config.device
    )[0].item()
    
    print(f"   Response: {response}")
    print(f"   Reward: {reward:.4f}")
    print()

print("="*70)
print(" Training and Evaluation Complete!")
print("="*70)
print(f"\n All outputs saved to: {config.output_dir}")
print(f"  - Final model: {os.path.join(config.output_dir, 'final')}")
print(f"  - Training curves: {os.path.join(config.output_dir, 'training_curves.png')}")
print(f"  - Training history: {os.path.join(config.output_dir, 'training_history.json')}")