"""
RLHF Project - Notebook 4: Evaluation & Comparison
Comprehensive evaluation comparing GPT-2 Small vs Medium after RLHF
"""

# Install required packages
# !pip install transformers datasets torch accelerate trl nltk rouge-score bert-score -q

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel
)
from datasets import load_from_disk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import json
from typing import List, Dict, Tuple
from collections import defaultdict

print("="*60)
print("RLHF Evaluation & Model Comparison")
print("="*60)

# Configuration
class EvalConfig:
    # Models to evaluate
    models = [
        {
            'name': 'gpt2',
            'display': 'GPT-2 Small (Baseline)',
            'path': 'gpt2',
            'type': 'baseline'
        },
        {
            'name': 'gpt2_ppo',
            'display': 'GPT-2 Small (PPO)',
            'path': './ppo_models/gpt2/final',
            'type': 'trained'
        },
        {
            'name': 'gpt2-medium',
            'display': 'GPT-2 Medium (Baseline)',
            'path': 'gpt2-medium',
            'type': 'baseline'
        },
        {
            'name': 'gpt2-medium_ppo',
            'display': 'GPT-2 Medium (PPO)',
            'path': './ppo_models/gpt2-medium/final',
            'type': 'trained'
        }
    ]
    
    # Reward model
    reward_model_path = "./reward_model_distilbert"
    reward_model_name = "distilbert-base-uncased"
    
    # Data
    data_path = "./rlhf_compact_data"
    num_eval_samples = 200
    
    # Generation
    max_new_tokens = 50
    temperature = 1.0
    top_k = 50
    top_p = 0.95
    num_generations_per_prompt = 3  # For diversity metrics
    
    # Output
    output_dir = "./evaluation_results"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = EvalConfig()

class RewardModel(nn.Module):
    """Reward model for scoring"""
    def __init__(self, model_path: str):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_path)
        self.reward_head = nn.Linear(self.transformer.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        reward = self.reward_head(hidden_state)
        return reward.squeeze(-1)

def load_models(config):
    """Load all models for evaluation"""
    print("\n Loading models...")
    
    models = {}
    tokenizers = {}
    
    for model_config in config.models:
        name = model_config['name']
        path = model_config['path']
        display = model_config['display']
        
        print(f"  Loading {display}...")
        
        try:
            # Load model
            if model_config['type'] == 'trained':
                # For PPO trained models, load from checkpoint
                from trl import AutoModelForCausalLMWithValueHead
                model = AutoModelForCausalLMWithValueHead.from_pretrained(path)
            else:
                # For baseline models
                model = AutoModelForCausalLM.from_pretrained(path)
            
            model.to(config.device)
            model.eval()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(path)
            tokenizer.pad_token = tokenizer.eos_token
            
            models[name] = model
            tokenizers[name] = tokenizer
            
        except Exception as e:
            print(f"    Warning: Could not load {display}: {e}")
            print(f"    Skipping this model...")
            continue
    
    # Load reward model
    print("  Loading reward model...")
    reward_model = RewardModel(config.reward_model_path)
    state_dict = torch.load(f"{config.reward_model_path}/best_model.pt", 
                           map_location=config.device)
    reward_model.load_state_dict(state_dict)
    reward_model.to(config.device)
    reward_model.eval()
    
    reward_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
    
    print(f" Loaded {len(models)} models successfully")
    
    return models, tokenizers, reward_model, reward_tokenizer

def generate_responses(prompt: str, model, tokenizer, config, num_samples: int = 1):
    """Generate responses for a prompt"""
    responses = []
    
    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        responses.append(response)
    
    return responses

def compute_reward_score(text: str, reward_model, reward_tokenizer, device):
    """Compute reward for a text"""
    encoded = reward_tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        reward = reward_model(
            encoded['input_ids'],
            encoded['attention_mask']
        ).item()
    
    return reward

def compute_diversity_metrics(responses: List[str]):
    """Compute diversity metrics for a set of responses"""
    if len(responses) <= 1:
        return {'unique_ratio': 1.0, 'avg_length': len(responses[0].split())}
    
    # Unique response ratio
    unique_responses = len(set(responses))
    unique_ratio = unique_responses / len(responses)
    
    # Average length
    avg_length = np.mean([len(r.split()) for r in responses])
    
    # Token diversity (unique tokens / total tokens)
    all_tokens = []
    for r in responses:
        all_tokens.extend(r.lower().split())
    
    token_diversity = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
    
    return {
        'unique_ratio': unique_ratio,
        'avg_length': avg_length,
        'token_diversity': token_diversity
    }

def evaluate_models(models, tokenizers, reward_model, reward_tokenizer, config):
    """Comprehensive model evaluation"""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Load test prompts
    dataset = load_from_disk(config.data_path)
    test_prompts = [ex['prompt'] for ex in dataset['validation'][:config.num_eval_samples]]
    
    print(f"\nEvaluating on {len(test_prompts)} prompts...")
    
    results = defaultdict(lambda: defaultdict(list))
    
    # Evaluate each model
    for model_name, model in models.items():
        tokenizer = tokenizers[model_name]
        model_display = next(m['display'] for m in config.models if m['name'] == model_name)
        
        print(f"\n Evaluating {model_display}...")
        
        for prompt in tqdm(test_prompts, desc=f"  {model_name}"):
            # Generate responses
            responses = generate_responses(
                prompt, model, tokenizer, config, 
                num_samples=config.num_generations_per_prompt
            )
            
            # Compute metrics for each response
            for response in responses:
                full_text = prompt + " " + response
                
                # Reward score
                reward = compute_reward_score(
                    full_text, reward_model, reward_tokenizer, config.device
                )
                results[model_name]['rewards'].append(reward)
                
                # Length
                response_length = len(response.split())
                results[model_name]['lengths'].append(response_length)
            
            # Diversity metrics
            diversity = compute_diversity_metrics(responses)
            results[model_name]['diversity_scores'].append(diversity)
    
    return results, test_prompts

def analyze_results(results, config):
    """Analyze and summarize results"""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    summary = {}
    
    for model_name, metrics in results.items():
        model_display = next(m['display'] for m in config.models if m['name'] == model_name)
        
        # Compute statistics
        mean_reward = np.mean(metrics['rewards'])
        std_reward = np.std(metrics['rewards'])
        mean_length = np.mean(metrics['lengths'])
        
        # Diversity
        diversity_scores = metrics['diversity_scores']
        mean_unique_ratio = np.mean([d['unique_ratio'] for d in diversity_scores])
        mean_token_div = np.mean([d['token_diversity'] for d in diversity_scores])
        
        summary[model_name] = {
            'display': model_display,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'unique_ratio': mean_unique_ratio,
            'token_diversity': mean_token_div
        }
        
        print(f"\n{model_display}:")
        print(f"  Mean Reward: {mean_reward:.4f} (±{std_reward:.4f})")
        print(f"  Mean Length: {mean_length:.2f} words")
        print(f"  Unique Response Ratio: {mean_unique_ratio:.3f}")
        print(f"  Token Diversity: {mean_token_div:.3f}")
    
    return summary

def create_comparison_plots(results, summary, config):
    """Create comprehensive comparison visualizations"""
    import os
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Prepare data
    model_names = list(summary.keys())
    display_names = [summary[m]['display'] for m in model_names]
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Mean Reward Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    rewards = [summary[m]['mean_reward'] for m in model_names]
    errors = [summary[m]['std_reward'] for m in model_names]
    colors = ['lightblue' if 'Small' in name else 'lightcoral' for name in display_names]
    ax1.bar(range(len(model_names)), rewards, yerr=errors, color=colors, alpha=0.7)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels([n.replace(' ', '\n') for n in display_names], fontsize=8)
    ax1.set_ylabel('Mean Reward Score')
    ax1.set_title('Average Reward Score Comparison')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Response Length Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    lengths = [summary[m]['mean_length'] for m in model_names]
    ax2.bar(range(len(model_names)), lengths, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels([n.replace(' ', '\n') for n in display_names], fontsize=8)
    ax2.set_ylabel('Mean Response Length (words)')
    ax2.set_title('Average Response Length')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Diversity Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(model_names))
    width = 0.35
    unique_ratios = [summary[m]['unique_ratio'] for m in model_names]
    token_divs = [summary[m]['token_diversity'] for m in model_names]
    ax3.bar(x - width/2, unique_ratios, width, label='Unique Ratio', alpha=0.7)
    ax3.bar(x + width/2, token_divs, width, label='Token Diversity', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([n.replace(' ', '\n') for n in display_names], fontsize=8)
    ax3.set_ylabel('Score')
    ax3.set_title('Diversity Metrics')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Reward Distribution (violin plot)
    ax4 = fig.add_subplot(gs[1, :])
    reward_data = [results[m]['rewards'] for m in model_names]
    parts = ax4.violinplot(reward_data, positions=range(len(model_names)), 
                           showmeans=True, showmedians=True)
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(display_names, rotation=15, ha='right')
    ax4.set_ylabel('Reward Score Distribution')
    ax4.set_title('Reward Score Distributions Across Models')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Scaling Analysis (Small vs Medium)
    ax5 = fig.add_subplot(gs[2, 0])
    baseline_small = next((m for m in model_names if 'gpt2' in m and 'medium' not in m 
                          and 'ppo' not in m), None)
    ppo_small = next((m for m in model_names if 'gpt2' in m and 'medium' not in m 
                     and 'ppo' in m), None)
    baseline_medium = next((m for m in model_names if 'medium' in m and 'ppo' not in m), None)
    ppo_medium = next((m for m in model_names if 'medium' in m and 'ppo' in m), None)
    
    if all([baseline_small, ppo_small, baseline_medium, ppo_medium]):
        categories = ['Baseline', 'After PPO']
        small_rewards = [summary[baseline_small]['mean_reward'], 
                        summary[ppo_small]['mean_reward']]
        medium_rewards = [summary[baseline_medium]['mean_reward'], 
                         summary[ppo_medium]['mean_reward']]
        
        x = np.arange(len(categories))
        width = 0.35
        ax5.bar(x - width/2, small_rewards, width, label='GPT-2 Small', color='lightblue', alpha=0.7)
        ax5.bar(x + width/2, medium_rewards, width, label='GPT-2 Medium', color='lightcoral', alpha=0.7)
        ax5.set_xticks(x)
        ax5.set_xticklabels(categories)
        ax5.set_ylabel('Mean Reward')
        ax5.set_title('Scaling Effect: Small vs Medium')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
    
    # 6. RLHF Improvement
    ax6 = fig.add_subplot(gs[2, 1])
    if all([baseline_small, ppo_small, baseline_medium, ppo_medium]):
        small_improvement = ((summary[ppo_small]['mean_reward'] - 
                             summary[baseline_small]['mean_reward']) / 
                            abs(summary[baseline_small]['mean_reward']) * 100)
        medium_improvement = ((summary[ppo_medium]['mean_reward'] - 
                              summary[baseline_medium]['mean_reward']) / 
                             abs(summary[baseline_medium]['mean_reward']) * 100)
        
        improvements = [small_improvement, medium_improvement]
        models_compared = ['Small', 'Medium']
        colors_imp = ['lightblue', 'lightcoral']
        ax6.bar(models_compared, improvements, color=colors_imp, alpha=0.7)
        ax6.set_ylabel('Reward Improvement (%)')
        ax6.set_title('RLHF Improvement by Model Size')
        ax6.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax6.grid(axis='y', alpha=0.3)
    
    # 7. Summary Table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('tight')
    ax7.axis('off')
    
    table_data = []
    for model_name in model_names:
        s = summary[model_name]
        table_data.append([
            s['display'].replace(' ', '\n'),
            f"{s['mean_reward']:.3f}",
            f"{s['mean_length']:.1f}",
            f"{s['token_diversity']:.3f}"
        ])
    
    table = ax7.table(cellText=table_data,
                     colLabels=['Model', 'Reward', 'Length', 'Diversity'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    plt.suptitle('RLHF Model Comparison: GPT-2 Small vs Medium', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f"{config.output_dir}/comprehensive_comparison.png", 
                dpi=300, bbox_inches='tight')
    print(f"\n Comprehensive comparison plot saved to {config.output_dir}")
    plt.show()

def generate_qualitative_examples(models, tokenizers, reward_model, reward_tokenizer, config):
    """Generate side-by-side qualitative comparison"""
    print("\n" + "="*60)
    print("QUALITATIVE EXAMPLES")
    print("="*60)
    
    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "How do I make scrambled eggs?",
        "Explain what machine learning is in simple terms.",
        "What are the benefits of exercise?",
        "Write a short poem about the moon."
    ]
    
    examples = []
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print('='*60)
        
        example = {'prompt': prompt, 'responses': {}}
        
        for model_name, model in models.items():
            tokenizer = tokenizers[model_name]
            model_display = next(m['display'] for m in config.models if m['name'] == model_name)
            
            # Generate response
            response = generate_responses(prompt, model, tokenizer, config, num_samples=1)[0]
            
            # Compute reward
            full_text = prompt + " " + response
            reward = compute_reward_score(full_text, reward_model, reward_tokenizer, config.device)
            
            example['responses'][model_name] = {
                'text': response,
                'reward': reward,
                'display': model_display
            }
            
            print(f"\n{model_display} (Reward: {reward:.4f}):")
            print(f"  {response}")
        
        examples.append(example)
    
    return examples

def save_results(summary, examples, config):
    """Save all results to files"""
    import os
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save summary statistics
    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_df.to_csv(f"{config.output_dir}/summary_statistics.csv")
    print(f"\n Summary statistics saved to {config.output_dir}/summary_statistics.csv")
    
    # Save qualitative examples
    with open(f"{config.output_dir}/qualitative_examples.json", 'w') as f:
        json.dump(examples, f, indent=2)
    print(f" Qualitative examples saved to {config.output_dir}/qualitative_examples.json")
    
    # Create markdown report
    report = "# RLHF Model Comparison Report\n\n"
    report += "## Summary Statistics\n\n"
    report += "| Model | Mean Reward | Std Reward | Mean Length | Token Diversity |\n"
    report += "|-------|-------------|------------|-------------|------------------|\n"
    
    for model_name, stats in summary.items():
        report += f"| {stats['display']} | {stats['mean_reward']:.4f} | {stats['std_reward']:.4f} | "
        report += f"{stats['mean_length']:.2f} | {stats['token_diversity']:.4f} |\n"
    
    report += "\n## Key Findings\n\n"
    
    # Calculate improvements
    baseline_small = next((m for m in summary.keys() if 'gpt2' in m and 'medium' not in m and 'ppo' not in m), None)
    ppo_small = next((m for m in summary.keys() if 'gpt2' in m and 'medium' not in m and 'ppo' in m), None)
    baseline_medium = next((m for m in summary.keys() if 'medium' in m and 'ppo' not in m), None)
    ppo_medium = next((m for m in summary.keys() if 'medium' in m and 'ppo' in m), None)
    
    if all([baseline_small, ppo_small, baseline_medium, ppo_medium]):
        small_imp = summary[ppo_small]['mean_reward'] - summary[baseline_small]['mean_reward']
        medium_imp = summary[ppo_medium]['mean_reward'] - summary[baseline_medium]['mean_reward']
        
        report += f"### RLHF Training Impact\n\n"
        report += f"- **GPT-2 Small**: Reward improvement of {small_imp:+.4f}\n"
        report += f"- **GPT-2 Medium**: Reward improvement of {medium_imp:+.4f}\n\n"
        
        report += f"### Model Size Comparison\n\n"
        report += f"- **Baseline**: Medium outperforms Small by "
        report += f"{summary[baseline_medium]['mean_reward'] - summary[baseline_small]['mean_reward']:+.4f}\n"
        report += f"- **After PPO**: Medium outperforms Small by "
        report += f"{summary[ppo_medium]['mean_reward'] - summary[ppo_small]['mean_reward']:+.4f}\n\n"
    
    report += "\n## Qualitative Examples\n\n"
    
    for i, example in enumerate(examples[:3], 1):  # First 3 examples
        report += f"### Example {i}\n\n"
        report += f"**Prompt**: {example['prompt']}\n\n"
        
        for model_name, response_data in example['responses'].items():
            report += f"**{response_data['display']}** (Reward: {response_data['reward']:.4f}):\n"
            report += f"> {response_data['text']}\n\n"
    
    with open(f"{config.output_dir}/evaluation_report.md", 'w') as f:
        f.write(report)
    
    print(f"💾 Evaluation report saved to {config.output_dir}/evaluation_report.md")

def compare_parameter_efficiency(summary, config):
    """Analyze parameter efficiency"""
    print("\n" + "="*60)
    print("PARAMETER EFFICIENCY ANALYSIS")
    print("="*60)
    
    # Parameter counts
    param_counts = {
        'gpt2': 124_000_000,
        'gpt2-medium': 355_000_000
    }
    
    baseline_small = next((m for m in summary.keys() if 'gpt2' in m and 'medium' not in m and 'ppo' not in m), None)
    ppo_small = next((m for m in summary.keys() if 'gpt2' in m and 'medium' not in m and 'ppo' in m), None)
    baseline_medium = next((m for m in summary.keys() if 'medium' in m and 'ppo' not in m), None)
    ppo_medium = next((m for m in summary.keys() if 'medium' in m and 'ppo' in m), None)
    
    if all([baseline_small, ppo_small, baseline_medium, ppo_medium]):
        # Reward per million parameters
        small_baseline_eff = summary[baseline_small]['mean_reward'] / (param_counts['gpt2'] / 1e6)
        small_ppo_eff = summary[ppo_small]['mean_reward'] / (param_counts['gpt2'] / 1e6)
        medium_baseline_eff = summary[baseline_medium]['mean_reward'] / (param_counts['gpt2-medium'] / 1e6)
        medium_ppo_eff = summary[ppo_medium]['mean_reward'] / (param_counts['gpt2-medium'] / 1e6)
        
        print("\nReward per Million Parameters:")
        print(f"  GPT-2 Small (Baseline):  {small_baseline_eff:.6f}")
        print(f"  GPT-2 Small (PPO):       {small_ppo_eff:.6f}")
        print(f"  GPT-2 Medium (Baseline): {medium_baseline_eff:.6f}")
        print(f"  GPT-2 Medium (PPO):      {medium_ppo_eff:.6f}")
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        models_labels = ['Small\n(Baseline)', 'Small\n(PPO)', 'Medium\n(Baseline)', 'Medium\n(PPO)']
        efficiencies = [small_baseline_eff, small_ppo_eff, medium_baseline_eff, medium_ppo_eff]
        colors = ['lightblue', 'blue', 'lightcoral', 'red']
        
        ax.bar(models_labels, efficiencies, color=colors, alpha=0.7)
        ax.set_ylabel('Reward Score per Million Parameters')
        ax.set_title('Parameter Efficiency Comparison')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{config.output_dir}/parameter_efficiency.png", dpi=150)
        print(f"\n Parameter efficiency plot saved")
        plt.show()

# Main execution
if __name__ == "__main__":
    import os
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("\n Starting comprehensive evaluation...\n")
    
    # Load all models
    models, tokenizers, reward_model, reward_tokenizer = load_models(config)
    
    if len(models) == 0:
        print(" No models loaded. Please check model paths.")
        exit(1)
    
    # Run quantitative evaluation
    results, test_prompts = evaluate_models(
        models, tokenizers, reward_model, reward_tokenizer, config
    )
    
    # Analyze results
    summary = analyze_results(results, config)
    
    # Create visualizations
    create_comparison_plots(results, summary, config)
    
    # Generate qualitative examples
    examples = generate_qualitative_examples(
        models, tokenizers, reward_model, reward_tokenizer, config
    )
    
    # Parameter efficiency analysis
    compare_parameter_efficiency(summary, config)
    
    # Save all results
    save_results(summary, examples, config)
    
    print("\n" + "="*60)
    print(" EVALUATION COMPLETE")
    print("="*60)
    print(f"\nAll results saved to: {config.output_dir}/")
    print("\nGenerated files:")
    print("  - comprehensive_comparison.png")
    print("  - parameter_efficiency.png")
    print("  - summary_statistics.csv")
    print("  - qualitative_examples.json")
    print("  - evaluation_report.md")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    best_model = max(summary.items(), key=lambda x: x[1]['mean_reward'])
    print(f"\n Best performing model: {best_model[1]['display']}")
    print(f"   Mean reward: {best_model[1]['mean_reward']:.4f}")
    
    print("\n Project complete! Check the evaluation report for detailed findings.")