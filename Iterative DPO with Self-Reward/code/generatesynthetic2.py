
# Generate Synthetic Preferences Round 2 - Fixed for Colab
# Run this AFTER DPO Round 1 completes

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from peft import PeftModel
from tqdm import tqdm
import json
import os
import torch.nn as nn

torch.manual_seed(43)
np.random.seed(43)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
class Config:
    sft_model_dir = "./outputs/sft_model"  # Base model for LoRA
    dpo_round1_dir = "./outputs/dpo_round1"  # LoRA adapters
    reward_model_dir = "./outputs/reward_model"
    synthetic_data_path2 = "./outputs/synthetic_preferences_round2.json"
    dataset_name = "Anthropic/hh-rlhf"
    num_gen_samples = 500  # Reduced from 1000 for faster generation (~25-30 min)
    max_length = 256
    num_responses_per_prompt = 2
    temperature = 0.85  # Slightly lower than Round 1 for more refined outputs
    top_p = 0.95

config = Config()









print(f"max length is {config.max_length}")
# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.dpo_round1_dir)
tokenizer.pad_token = tokenizer.eos_token

# Load DPO Round 1 model (Base + LoRA adapters)
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    config.sft_model_dir,
    torch_dtype=torch.float16,  # fp16 for Colab T4
    device_map="auto",
    low_cpu_mem_usage=True
)

print("Loading DPO Round 1 LoRA adapters...")
dpo_model = PeftModel.from_pretrained(base_model, config.dpo_round1_dir)
dpo_model.eval()

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

# Convert reward head to fp16
reward_model.reward_head = reward_model.reward_head.to(torch.float16)
reward_model = reward_model.to(device)
reward_model.eval()

print("Model dtype check:")
print(f"DPO model dtype: {next(dpo_model.parameters()).dtype}")
print(f"Reward head dtype: {next(reward_model.reward_head.parameters()).dtype}")

# Load NEW prompts (different from Round 1)
print("\nLoading prompts...")
dataset = load_dataset(config.dataset_name, split="train")
# Skip first 1000 used in Round 1, get next 500
start_idx = 1000
end_idx = start_idx + config.num_gen_samples
dataset = dataset.select(range(start_idx, min(end_idx, len(dataset))))

prompts = []
for ex in dataset:
    try:
        prompt = ex['chosen'].split('Assistant:')[0].replace('Human:', '').strip()
        if prompt and len(prompt) > 10:
            prompts.append(prompt)
    except:
        continue

prompts = prompts[:config.num_gen_samples]
print(f"Number of prompts: {len(prompts)}")

# Generate responses and score
synthetic_data = []

for prompt in tqdm(prompts, desc="Generating Round 2 responses"):
    responses = []

    # Generate multiple responses
    for _ in range(config.num_responses_per_prompt):
        input_text = f"Human: {prompt}\n\nAssistant:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=config.max_length).to(device)

        with torch.no_grad():
            outputs = dpo_model.generate(
                **inputs,
                max_new_tokens=100,  # Reduced from 150 for faster generation
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("Assistant:")[-1].strip()
        responses.append(response)

    # Score responses with reward model
    scores = []
    for response in responses:
        full_text = f"Human: {prompt}\n\nAssistant: {response}"
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=config.max_length, padding="max_length").to(device)

        with torch.no_grad():
            score = reward_model(inputs["input_ids"], inputs["attention_mask"])

        scores.append(score.item())

    # Select chosen and rejected
    max_idx = np.argmax(scores)
    min_idx = np.argmin(scores)

    if max_idx != min_idx and abs(scores[max_idx] - scores[min_idx]) > 0.1:
        synthetic_data.append({
            "prompt": prompt,
            "chosen": responses[max_idx],
            "rejected": responses[min_idx],
            "chosen_score": float(scores[max_idx]),
            "rejected_score": float(scores[min_idx])
        })

print(f"\nGenerated {len(synthetic_data)} synthetic preference pairs for Round 2")

# Save synthetic data
with open(config.synthetic_data_path2, 'w') as f:
    json.dump(synthetic_data, f, indent=2)

print(f" Synthetic preferences Round 2 saved to {config.synthetic_data_path2}")
print("\n Synthetic Data Generation Round 2 Complete!")

# Print statistics
if synthetic_data:
    chosen_scores = [d['chosen_score'] for d in synthetic_data]
    rejected_scores = [d['rejected_score'] for d in synthetic_data]
    score_diffs = [d['chosen_score'] - d['rejected_score'] for d in synthetic_data]

    print("\n=== Round 2 Statistics ===")
    print(f"Average chosen score: {np.mean(chosen_scores):.4f}")
    print(f"Average rejected score: {np.mean(rejected_scores):.4f}")
    print(f"Average score difference: {np.mean(score_diffs):.4f}")
    print(f"Min score difference: {np.min(score_diffs):.4f}")
    print(f"Max score difference: {np.max(score_diffs):.4f}")