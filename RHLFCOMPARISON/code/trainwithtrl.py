"""
RLHF Project - Notebook 3: PPO Training (Colab Optimized)
Trains GPT-2 Small and Medium using PPO with the trained reward model
Memory-optimized to prevent crashes
"""

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================



import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModel
)
from datasets import load_from_disk
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
import os
import gc
from typing import List, Dict

print("="*60)
print("RLHF PPO Training - GPT-2 Comparison (Colab Optimized)")
print("="*60)

# Check GPU
if torch.cuda.is_available():
    print(f" GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(" No GPU available - training will be very slow")

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Optimized config for Colab (15GB RAM)"""
    
    # Models to compare
    models_to_train = [
        {"name": "gpt2", "display": "GPT-2 Small (124M)", "batch_size": 8},
        {"name": "gpt2-medium", "display": "GPT-2 Medium (355M)", "batch_size": 4}
    ]
    
    # Paths (UPDATE THESE WITH YOUR PATHS)
    reward_model_path = "./reward_model_distilgpt2"  # Path to trained reward model
    reward_model_name = "distilgpt2"
    data_path = "./rlhf_compact_data"  # Path to processed dataset
    output_base = "./ppo_models"
    
    # Training parameters (optimized for Colab)
    max_samples_per_model = 500  # Reduced for faster training
    mini_batch_size = 1
    gradient_accumulation_steps = 4
    ppo_epochs = 2  # Reduced from 4
    learning_rate = 1.41e-5
    
    # Generation parameters
    max_new_tokens = 40  # Reduced from 50
    temperature = 1.0
    top_k = 50
    top_p = 0.95
    
    # Training steps
    num_training_steps = 100  # Reduced from 200 for faster completion
    save_freq = 50
    eval_freq = 25
    
    # Memory optimization
    use_fp16 = True  # Use mixed precision
    clear_cache_freq = 10  # Clear cache every N steps
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = TrainingConfig()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_memory():
    """Clear GPU and Python memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# ============================================================================
# REWARD MODEL
# ============================================================================

class RewardModel(nn.Module):
    """Load the trained reward model"""
    def __init__(self, model_path: str):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_path)
        self.reward_head = nn.Linear(self.transformer.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # For GPT models: use LAST token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        hidden_state = outputs.last_hidden_state[
            range(len(sequence_lengths)), sequence_lengths
        ]
        
        reward = self.reward_head(hidden_state)
        return reward.squeeze(-1)

def load_reward_model(model_path: str, device: str):
    """Load trained reward model"""
    print(f"\ Loading reward model from {model_path}...")
    
    reward_model = RewardModel(model_path)
    
    # Load trained weights
    state_dict = torch.load(
        f"{model_path}/best_model.pt", 
        map_location=device
    )
    reward_model.load_state_dict(state_dict)
    reward_model.to(device)
    
    if config.use_fp16:
        reward_model.half()
    
    reward_model.eval()
    
    print(" Reward model loaded successfully")
    return reward_model

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_prompts(dataset, num_samples: int):
    """Extract prompts from dataset"""
    prompts = []
    
    for example in dataset['train']:
        if len(prompts) >= num_samples:
            break
        prompts.append(example['prompt'])
    
    print(f" Loaded {len(prompts)} prompts")
    return prompts

def compute_rewards(
    texts: List[str], 
    reward_model, 
    reward_tokenizer, 
    device: str,
    batch_size: int = 8
):
    """Compute rewards for generated texts (batched)"""
    rewards = []
    
    reward_model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encoded = reward_tokenizer(
                batch_texts,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(device)
            
            batch_rewards = reward_model(
                encoded['input_ids'],
                encoded['attention_mask']
            )
            
            rewards.extend(batch_rewards.cpu().tolist())
    
    return rewards

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_ppo_model(model_config: Dict, config: TrainingConfig):
    """Train a single model with PPO"""
    model_name = model_config['name']
    display_name = model_config['display']
    batch_size = model_config['batch_size']
    
    print("\n" + "="*60)
    print(f" Training {display_name}")
    print("="*60)
    
    # Clear memory before starting
    clear_memory()
    
    # Create output directory
    output_dir = f"{config.output_base}/{model_name.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # LOAD MODELS
    # ========================================================================
    
    print("\n Loading tokenizers and models...")
    
    # Policy tokenizer
    policy_tokenizer = AutoTokenizer.from_pretrained(model_name)
    policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.padding_side = 'left'
    
    # Reward tokenizer
    reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    # Load reward model
    reward_model = load_reward_model(config.reward_model_path, config.device)
    
    # Load policy model with value head
    print(f"Loading policy model: {model_name}...")
    
    if config.use_fp16:
        policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
    else:
        policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    policy_model.to(config.device)
    
    # Load reference model (frozen)
    print("Loading reference model...")
    if config.use_fp16:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    ref_model.to(config.device)
    ref_model.eval()
    
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    
    print(f" Models loaded")
    print(f"  Policy parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    print_memory_usage()
    
    # ========================================================================
    # PPO CONFIGURATION
    # ========================================================================
    
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=config.learning_rate,
        batch_size=batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        ppo_epochs=config.ppo_epochs,
        optimize_cuda_cache=True,
    )
    
    # Initialize PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=policy_tokenizer,
    )
    
    # ========================================================================
    # LOAD DATASET
    # ========================================================================
    
    print("\n Loading prompts...")
    dataset = load_from_disk(config.data_path)
    prompts = prepare_prompts(dataset, config.max_samples_per_model)
    
    # Generation kwargs
    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "do_sample": True,
        "pad_token_id": policy_tokenizer.pad_token_id,
    }
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    print("\n Starting PPO training...")
    print(f"  Total steps: {config.num_training_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Save frequency: {config.save_freq}")
    
    history = {
        'rewards': [],
        'policy_loss': [],
        'value_loss': [],
        'kl_div': [],
        'response_lengths': []
    }
    
    step = 0
    prompt_idx = 0
    
    progress_bar = tqdm(total=config.num_training_steps, desc="PPO Steps")
    
    try:
        while step < config.num_training_steps:
            # Get batch of prompts
            batch_prompts = []
            for _ in range(batch_size):
                batch_prompts.append(prompts[prompt_idx % len(prompts)])
                prompt_idx += 1
            
            # Tokenize prompts
            prompt_tensors = []
            for prompt in batch_prompts:
                encoded = policy_tokenizer.encode(
                    prompt, 
                    return_tensors="pt"
                )[0].to(config.device)
                prompt_tensors.append(encoded)
            
            # Generate responses
            response_tensors = []
            with torch.no_grad():
                for prompt_tensor in prompt_tensors:
                    response = policy_model.generate(
                        prompt_tensor.unsqueeze(0),
                        **generation_kwargs
                    )
                    # Extract only the new tokens
                    response_tensors.append(response.squeeze()[len(prompt_tensor):])
            
            # Decode responses
            batch_responses = [
                policy_tokenizer.decode(r, skip_special_tokens=True)
                for r in response_tensors
            ]
            
            # Compute rewards
            full_texts = [p + " " + r for p, r in zip(batch_prompts, batch_responses)]
            rewards = compute_rewards(
                full_texts, 
                reward_model, 
                reward_tokenizer, 
                config.device,
                batch_size=4
            )
            rewards_tensor = [torch.tensor(r, device=config.device) for r in rewards]
            
            # Run PPO step
            stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards_tensor)
            
            # Log metrics
            mean_reward = np.mean(rewards)
            mean_length = np.mean([len(r.split()) for r in batch_responses])
            
            history['rewards'].append(mean_reward)
            history['policy_loss'].append(stats.get('ppo/loss/policy', 0))
            history['value_loss'].append(stats.get('ppo/loss/value', 0))
            history['kl_div'].append(stats.get('objective/kl', 0))
            history['response_lengths'].append(mean_length)
            
            # Update progress
            progress_bar.update(1)
            progress_bar.set_postfix({
                'reward': f'{mean_reward:.3f}',
                'kl': f'{history["kl_div"][-1]:.4f}',
                'len': f'{mean_length:.1f}'
            })
            
            step += 1
            
            # Clear cache periodically
            if step % config.clear_cache_freq == 0:
                clear_memory()
            
            # Evaluation logging
            if step % config.eval_freq == 0:
                print(f"\n Step {step} Metrics:")
                print(f"  Avg Reward (last 10): {np.mean(history['rewards'][-10:]):.4f}")
                print(f"  Avg KL Div (last 10): {np.mean(history['kl_div'][-10:]):.4f}")
                print(f"  Sample response: {batch_responses[0][:100]}...")
                print_memory_usage()
            
            # Save checkpoint
            if step % config.save_freq == 0 and step > 0:
                checkpoint_dir = f"{output_dir}/checkpoint_{step}"
                print(f"\n Saving checkpoint at step {step}...")
                policy_model.save_pretrained(checkpoint_dir)
                policy_tokenizer.save_pretrained(checkpoint_dir)
                
                # Save history
                with open(f"{output_dir}/training_history.json", 'w') as f:
                    json.dump(history, f, indent=2)
    
    except Exception as e:
        print(f"\n Error during training: {str(e)}")
        print("Saving current progress...")
        # Save what we have so far
        emergency_dir = f"{output_dir}/emergency_checkpoint"
        os.makedirs(emergency_dir, exist_ok=True)
        policy_model.save_pretrained(emergency_dir)
        policy_tokenizer.save_pretrained(emergency_dir)
        raise e
    
    finally:
        progress_bar.close()
    
    # ========================================================================
    # SAVE FINAL MODEL
    # ========================================================================
    
    print(f"\n Saving final model to {output_dir}/final...")
    policy_model.save_pretrained(f"{output_dir}/final")
    policy_tokenizer.save_pretrained(f"{output_dir}/final")
    
    # Save training history
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save summary statistics
    summary = {
        'model_name': model_name,
        'display_name': display_name,
        'total_steps': step,
        'final_reward': float(history['rewards'][-1]),
        'avg_reward_last_50': float(np.mean(history['rewards'][-50:])),
        'final_kl': float(history['kl_div'][-1]),
        'avg_kl_last_50': float(np.mean(history['kl_div'][-50:]))
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Training complete for {display_name}")
    print(f"\n Final Statistics:")
    print(f"  Final Reward: {summary['final_reward']:.4f}")
    print(f"  Avg Reward (last 50): {summary['avg_reward_last_50']:.4f}")
    print(f"  Final KL Div: {summary['final_kl']:.4f}")
    
    # Cleanup
    del policy_model, ref_model, reward_model, ppo_trainer
    clear_memory()
    
    return history, output_dir, summary

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_metrics(history: Dict, model_name: str, output_dir: str):
    """Plot training metrics for a single model"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards
    axes[0, 0].plot(history['rewards'], alpha=0.6, label='Rewards')
    axes[0, 0].plot(
        np.convolve(history['rewards'], np.ones(10)/10, mode='valid'),
        label='Moving Avg (10)', linewidth=2
    )
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title(f'{model_name} - Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # KL Divergence
    axes[0, 1].plot(history['kl_div'], alpha=0.8)
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].set_title(f'{model_name} - KL Divergence')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Policy Loss
    axes[1, 0].plot(history['policy_loss'], alpha=0.8)
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Policy Loss')
    axes[1, 0].set_title(f'{model_name} - Policy Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Response Lengths
    axes[1, 1].plot(history['response_lengths'], alpha=0.8)
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Avg Response Length (words)')
    axes[1, 1].set_title(f'{model_name} - Response Length')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_metrics.png", dpi=150, bbox_inches='tight')
    print(f"✓ Metrics plot saved to {output_dir}/training_metrics.png")
    plt.show()

def plot_comparison(histories: Dict[str, Dict], config: TrainingConfig):
    """Plot training comparison between models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (model_name, history) in enumerate(histories.items()):
        color = colors[idx % len(colors)]
        
        # Rewards with moving average
        axes[0, 0].plot(
            history['rewards'], 
            label=f'{model_name} (raw)', 
            color=color, 
            alpha=0.3
        )
        if len(history['rewards']) >= 10:
            moving_avg = np.convolve(
                history['rewards'], 
                np.ones(10)/10, 
                mode='valid'
            )
            axes[0, 0].plot(
                moving_avg,
                label=f'{model_name} (MA-10)',
                color=color,
                linewidth=2
            )
        
        # KL Divergence
        axes[0, 1].plot(
            history['kl_div'], 
            label=model_name,
            color=color,
            alpha=0.7
        )
        
        # Policy Loss
        axes[1, 0].plot(
            history['policy_loss'], 
            label=model_name,
            color=color,
            alpha=0.7
        )
        
        # Response Lengths
        axes[1, 1].plot(
            history['response_lengths'],
            label=model_name,
            color=color,
            alpha=0.7
        )
    
    # Configure subplots
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Reward Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].set_title('KL Divergence Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Policy Loss')
    axes[1, 0].set_title('Policy Loss Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Training Step')
    axes[1, 1].set_ylabel('Avg Response Length')
    axes[1, 1].set_title('Response Length Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.output_base}/model_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n Comparison plot saved to {config.output_base}/model_comparison.png")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n Starting PPO training for model comparison...\n")
    
    os.makedirs(config.output_base, exist_ok=True)
    
    histories = {}
    model_paths = {}
    summaries = {}
    
    # Train each model
    for model_config in config.models_to_train:
        try:
            print(f"\n{'='*60}")
            print(f"Training {model_config['display']}")
            print(f"{'='*60}")
            
            history, output_dir, summary = train_ppo_model(model_config, config)
            
            # Store results
            model_display = model_config['display']
            histories[model_display] = history
            model_paths[model_display] = output_dir
            summaries[model_display] = summary
            
            # Plot individual model metrics
            plot_training_metrics(history, model_display, output_dir)
            
            print(f"\n {model_display} training complete!")
            
        except Exception as e:
            print(f"\n Error training {model_config['display']}: {str(e)}")
            print("Continuing with next model...")
            continue
    
    # Plot comparison if we have multiple models
    if len(histories) > 1:
        print("\n Creating comparison plots...")
        plot_comparison(histories, config)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*60)
    print(" TRAINING SUMMARY")
    print("="*60)
    
    for model_name, summary in summaries.items():
        print(f"\n{model_name}:")
        print(f"  Total Steps: {summary['total_steps']}")
        print(f"  Final Reward: {summary['final_reward']:.4f}")
        print(f"  Avg Reward (last 50): {summary['avg_reward_last_50']:.4f}")
        print(f"  Final KL Div: {summary['final_kl']:.4f}")
        print(f"  Model Path: {model_paths[model_name]}/final")
    
    # Save overall summary
    overall_summary = {
        'models_trained': list(summaries.keys()),
        'model_summaries': summaries,
        'config': {
            'num_steps': config.num_training_steps,
            'learning_rate': config.learning_rate,
            'batch_sizes': {m['display']: m['batch_size'] for m in config.models_to_train}
        }
    }
    
    with open(f"{config.output_base}/overall_summary.json", 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    print(f"\n All models trained successfully!")
    print(f" Models saved to: {config.output_base}")
    print(f" Summary saved to: {config.output_base}/overall_summary.json")
    print("\n Next: Run Notebook 4 for comprehensive evaluation and comparison")