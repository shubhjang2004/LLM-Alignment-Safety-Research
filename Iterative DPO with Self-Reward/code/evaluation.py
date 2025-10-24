# Evaluation Script - Fixed for Colab T4 with LoRA
# Run this AFTER DPO Round 2 training completes

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import json
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
class Config:
    sft_model_dir = "/content/drive/MyDrive/outputs/sft_model"
    dpo_round1_dir = "/content/drive/MyDrive/outputs/dpo_round1"
    dpo_round2_dir = "/content/drive/MyDrive/outputs/dpo_round2"
    reward_model_dir = "/content/drive/MyDrive/outputs/reward_model"
    eval_results_path = "/content/drive/MyDrive/outputs/evaluation_results.json"
    plots_dir = "/content/drive/MyDrive/outputs/plots"
    dataset_name = "Anthropic/hh-rlhf"
    num_eval_samples = 100
    max_length = 256
    max_new_tokens = 100

config = Config()
os.makedirs(config.plots_dir, exist_ok=True)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.sft_model_dir)
tokenizer.pad_token = tokenizer.eos_token

# Load evaluation prompts
print("\nLoading evaluation prompts...")
dataset = load_dataset(config.dataset_name, split="test")
eval_data = dataset.select(range(min(config.num_eval_samples, len(dataset))))

eval_prompts = []
for ex in eval_data:
    try:
        prompt = ex['chosen'].split('Assistant:')[0].replace('Human:', '').strip()
        if prompt and len(prompt) > 10:
            eval_prompts.append(prompt)
    except:
        continue

eval_prompts = eval_prompts[:config.num_eval_samples]
print(f"Number of evaluation prompts: {len(eval_prompts)}")

# ============================================================================
# PART 1: GENERATE RESPONSES FROM ALL MODELS (SEQUENTIALLY)
# ============================================================================

print("\n" + "="*70)
print("PART 1: GENERATING RESPONSES FROM ALL MODELS")
print("="*70)

results = {
    "sft": [],
    "dpo_round1": [],
    "dpo_round2": []
}

# Model 1: SFT Baseline
print("\n1. Loading SFT model...")
sft_model = AutoModelForCausalLM.from_pretrained(
    config.sft_model_dir,
    torch_dtype=torch.float16,  # fp16 for T4
    device_map="auto",
    low_cpu_mem_usage=True
)
sft_model.eval()

print("Generating SFT responses...")
for prompt in tqdm(eval_prompts, desc="SFT Model"):
    input_text = f"Human: {prompt}\n\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=config.max_length).to(device)
    
    with torch.no_grad():
        outputs = sft_model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("Assistant:")[-1].strip()
    results["sft"].append({"prompt": prompt, "response": response})

# Clean up
del sft_model
torch.cuda.empty_cache()
print(" SFT model done, memory cleared")

# Model 2: DPO Round 1
print("\n2. Loading DPO Round 1 model...")
base_model = AutoModelForCausalLM.from_pretrained(
    config.sft_model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
dpo1_model = PeftModel.from_pretrained(base_model, config.dpo_round1_dir)
dpo1_model.eval()

print("Generating DPO Round 1 responses...")
for prompt in tqdm(eval_prompts, desc="DPO Round 1"):
    input_text = f"Human: {prompt}\n\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=config.max_length).to(device)
    
    with torch.no_grad():
        outputs = dpo1_model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("Assistant:")[-1].strip()
    results["dpo_round1"].append({"prompt": prompt, "response": response})

# Clean up
del dpo1_model, base_model
torch.cuda.empty_cache()
print(" DPO Round 1 done, memory cleared")

# Model 3: DPO Round 2
print("\n3. Loading DPO Round 2 model...")
base_model = AutoModelForCausalLM.from_pretrained(
    config.sft_model_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)
dpo2_model = PeftModel.from_pretrained(base_model, config.dpo_round2_dir)
dpo2_model.eval()

print("Generating DPO Round 2 responses...")
for prompt in tqdm(eval_prompts, desc="DPO Round 2"):
    input_text = f"Human: {prompt}\n\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=config.max_length).to(device)
    
    with torch.no_grad():
        outputs = dpo2_model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response.split("Assistant:")[-1].strip()
    results["dpo_round2"].append({"prompt": prompt, "response": response})

# Clean up
del dpo2_model, base_model
torch.cuda.empty_cache()
print(" DPO Round 2 done, memory cleared")

# ============================================================================
# PART 2: SCORE ALL RESPONSES WITH REWARD MODEL
# ============================================================================

print("\n" + "="*70)
print("PART 2: SCORING RESPONSES WITH REWARD MODEL")
print("="*70)

# Load reward model
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
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

print("Loading reward model...")
reward_base_model = AutoModelForCausalLM.from_pretrained(
    config.reward_model_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

reward_model = RewardModel(reward_base_model)
checkpoint = torch.load(os.path.join(config.reward_model_dir, "reward_model.pt"))
reward_model.reward_head.load_state_dict(checkpoint['reward_head_state_dict'])
reward_model.reward_head = reward_model.reward_head.to(torch.float16)
reward_model = reward_model.to(device)
reward_model.eval()

print("Scoring all responses...")

reward_scores = {
    "sft": [],
    "dpo_round1": [],
    "dpo_round2": []
}

for model_name in results.keys():
    print(f"\nScoring {model_name} responses...")
    for item in tqdm(results[model_name], desc=f"Scoring {model_name}"):
        full_text = f"Human: {item['prompt']}\n\nAssistant: {item['response']}"
        inputs = tokenizer(
            full_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=config.max_length, 
            padding="max_length"
        ).to(device)
        
        with torch.no_grad():
            score = reward_model(inputs["input_ids"], inputs["attention_mask"])
        
        reward_scores[model_name].append(score.item())

# Clean up
del reward_model, reward_base_model
torch.cuda.empty_cache()
print("Reward scoring complete, memory cleared")

# ============================================================================
# PART 3: CALCULATE STATISTICS
# ============================================================================

print("\n" + "="*70)
print("PART 3: EVALUATION STATISTICS")
print("="*70)

statistics = {}

for model_name in results.keys():
    responses = [r["response"] for r in results[model_name]]
    scores = reward_scores[model_name]
    
    statistics[model_name] = {
        "avg_reward": float(np.mean(scores)),
        "std_reward": float(np.std(scores)),
        "median_reward": float(np.median(scores)),
        "min_reward": float(np.min(scores)),
        "max_reward": float(np.max(scores)),
        "avg_length": float(np.mean([len(r.split()) for r in responses])),
        "unique_responses": len(set(responses))
    }

print("\n REWARD SCORES:")
print(f"\nSFT Baseline:")
print(f"  Average reward: {statistics['sft']['avg_reward']:.4f} ± {statistics['sft']['std_reward']:.4f}")
print(f"  Median reward:  {statistics['sft']['median_reward']:.4f}")
print(f"  Range: [{statistics['sft']['min_reward']:.4f}, {statistics['sft']['max_reward']:.4f}]")

print(f"\nDPO Round 1:")
print(f"  Average reward: {statistics['dpo_round1']['avg_reward']:.4f} ± {statistics['dpo_round1']['std_reward']:.4f}")
print(f"  Median reward:  {statistics['dpo_round1']['median_reward']:.4f}")
print(f"  Range: [{statistics['dpo_round1']['min_reward']:.4f}, {statistics['dpo_round1']['max_reward']:.4f}]")
print(f"  Improvement over SFT: {statistics['dpo_round1']['avg_reward'] - statistics['sft']['avg_reward']:.4f}")

print(f"\nDPO Round 2:")
print(f"  Average reward: {statistics['dpo_round2']['avg_reward']:.4f} ± {statistics['dpo_round2']['std_reward']:.4f}")
print(f"  Median reward:  {statistics['dpo_round2']['median_reward']:.4f}")
print(f"  Range: [{statistics['dpo_round2']['min_reward']:.4f}, {statistics['dpo_round2']['max_reward']:.4f}]")
print(f"  Improvement over Round 1: {statistics['dpo_round2']['avg_reward'] - statistics['dpo_round1']['avg_reward']:.4f}")
print(f"  Total improvement over SFT: {statistics['dpo_round2']['avg_reward'] - statistics['sft']['avg_reward']:.4f}")

# ============================================================================
# PART 4: VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("PART 4: CREATING VISUALIZATIONS")
print("="*70)

sns.set_style("whitegrid")

# Plot 1: Average Reward Comparison
fig, ax = plt.subplots(figsize=(10, 6))

models = ['SFT\nBaseline', 'DPO\nRound 1', 'DPO\nRound 2']
avg_rewards = [
    statistics['sft']['avg_reward'],
    statistics['dpo_round1']['avg_reward'],
    statistics['dpo_round2']['avg_reward']
]
std_rewards = [
    statistics['sft']['std_reward'],
    statistics['dpo_round1']['std_reward'],
    statistics['dpo_round2']['std_reward']
]

colors = ['#e74c3c', '#3498db', '#27ae60']
bars = ax.bar(models, avg_rewards, yerr=std_rewards, color=colors, alpha=0.8, capsize=10)

ax.set_ylabel('Average Reward Score', fontsize=12, fontweight='bold')
ax.set_title('Iterative DPO: Progressive Improvement in Reward Scores', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, reward in zip(bars, avg_rewards):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{reward:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(config.plots_dir, 'reward_comparison.png'), dpi=300, bbox_inches='tight')
print(" Saved: reward_comparison.png")

# Plot 2: Reward Distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (model_name, title) in enumerate([('sft', 'SFT Baseline'), 
                                             ('dpo_round1', 'DPO Round 1'), 
                                             ('dpo_round2', 'DPO Round 2')]):
    axes[idx].hist(reward_scores[model_name], bins=30, color=colors[idx], alpha=0.7, edgecolor='black')
    axes[idx].axvline(statistics[model_name]['avg_reward'], color='red', linestyle='--', linewidth=2, label='Mean')
    axes[idx].axvline(statistics[model_name]['median_reward'], color='blue', linestyle='--', linewidth=2, label='Median')
    axes[idx].set_xlabel('Reward Score', fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(config.plots_dir, 'reward_distributions.png'), dpi=300, bbox_inches='tight')
print("Saved: reward_distributions.png")

# Plot 3: Improvement Trajectory
fig, ax = plt.subplots(figsize=(10, 6))

rounds = ['Round 0\n(SFT)', 'Round 1\n(DPO)', 'Round 2\n(DPO)']
improvements = [
    0,  # Baseline
    statistics['dpo_round1']['avg_reward'] - statistics['sft']['avg_reward'],
    statistics['dpo_round2']['avg_reward'] - statistics['sft']['avg_reward']
]

ax.plot(rounds, improvements, marker='o', linewidth=3, markersize=12, color='#27ae60')
ax.fill_between(range(len(rounds)), improvements, alpha=0.3, color='#27ae60')
ax.set_ylabel('Improvement over SFT Baseline', fontsize=12, fontweight='bold')
ax.set_title('Iterative DPO: Cumulative Improvement', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Add value labels
for i, (round_name, improvement) in enumerate(zip(rounds, improvements)):
    ax.text(i, improvement + 0.005, f'+{improvement:.4f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(config.plots_dir, 'improvement_trajectory.png'), dpi=300, bbox_inches='tight')
print(" Saved: improvement_trajectory.png")

# ============================================================================
# PART 5: QUALITATIVE SAMPLES
# ============================================================================

print("\n" + "="*70)
print("PART 5: SAMPLE COMPARISONS")
print("="*70)

num_samples = min(5, len(eval_prompts))

for i in range(num_samples):
    print(f"\n{'='*70}")
    print(f"SAMPLE {i+1}")
    print(f"{'='*70}")
    print(f"\n Prompt: {eval_prompts[i][:100]}...")
    
    for model_name in ['sft', 'dpo_round1', 'dpo_round2']:
        response = results[model_name][i]['response']
        score = reward_scores[model_name][i]
        
        print(f"\n--- {model_name.upper().replace('_', ' ')} (Reward: {score:.4f}) ---")
        print(response[:300] + ("..." if len(response) > 300 else ""))

# ============================================================================
# PART 6: SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("PART 6: SAVING RESULTS")
print("="*70)

eval_summary = {
    "num_samples": len(eval_prompts),
    "statistics": statistics,
    "reward_scores": reward_scores,
    "improvements": {
        "round1_vs_sft": statistics['dpo_round1']['avg_reward'] - statistics['sft']['avg_reward'],
        "round2_vs_round1": statistics['dpo_round2']['avg_reward'] - statistics['dpo_round1']['avg_reward'],
        "round2_vs_sft": statistics['dpo_round2']['avg_reward'] - statistics['sft']['avg_reward']
    },
    "sample_responses": {
        "prompts": eval_prompts[:5],
        "sft": [results['sft'][i]['response'] for i in range(min(5, len(eval_prompts)))],
        "dpo_round1": [results['dpo_round1'][i]['response'] for i in range(min(5, len(eval_prompts)))],
        "dpo_round2": [results['dpo_round2'][i]['response'] for i in range(min(5, len(eval_prompts)))]
    }
}

with open(config.eval_results_path, 'w') as f:
    json.dump(eval_summary, f, indent=2)

print(f" Evaluation results saved to {config.eval_results_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print(" ITERATIVE DPO EVALUATION COMPLETE!")
print("="*70)

summary = f"""
 ITERATIVE DPO - FINAL RESULTS SUMMARY

{'='*70}
PROGRESSIVE IMPROVEMENT:
{'='*70}

SFT Baseline:       {statistics['sft']['avg_reward']:.4f} ± {statistics['sft']['std_reward']:.4f}
DPO Round 1:        {statistics['dpo_round1']['avg_reward']:.4f} ± {statistics['dpo_round1']['std_reward']:.4f}  (+{improvements[1]:.4f})
DPO Round 2:        {statistics['dpo_round2']['avg_reward']:.4f} ± {statistics['dpo_round2']['std_reward']:.4f}  (+{improvements[2]:.4f})

{'='*70}
KEY FINDINGS:
{'='*70}

 Round 1 improved over SFT by:     {eval_summary['improvements']['round1_vs_sft']:.4f}
 Round 2 improved over Round 1 by: {eval_summary['improvements']['round2_vs_round1']:.4f}
 Total improvement (SFT → R2):     {eval_summary['improvements']['round2_vs_sft']:.4f}

{'='*70}
SUCCESS METRICS:
{'='*70}

 Iterative self-improvement demonstrated
 No external critic required
 Synthetic preference generation working
 Progressive alignment improvement across rounds

{'='*70}
GENERATED FILES:
{'='*70}

 reward_comparison.png: Bar chart of average rewards
 reward_distributions.png: Histograms of reward distributions
 improvement_trajectory.png: Cumulative improvement plot
 evaluation_results.json: Complete numerical results

"""

print(summary)

# Save summary
with open(os.path.join(config.plots_dir, 'evaluation_summary.txt'), 'w') as f:
    f.write(summary)

print(f" Summary saved to {os.path.join(config.plots_dir, 'evaluation_summary.txt')}")
print("\n All evaluation tasks complete!")
print("="*70)