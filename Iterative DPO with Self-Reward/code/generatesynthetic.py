# Generate Synthetic Preferences - Fixed Version
# Run this AFTER reward model training

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import json
import os
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
class Config:
    sft_model_dir = "./outputs/sft_model"
    reward_model_dir = "./outputs/reward_model"
    synthetic_data_path = "./outputs/synthetic_preferences.json"
    dataset_name = "Anthropic/hh-rlhf"
    num_gen_samples = 1000
    max_length = 256
    num_responses_per_prompt = 2
    temperature = 0.9
    top_p = 0.95

config = Config()

# Load models and tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config.sft_model_dir)
tokenizer.pad_token = tokenizer.eos_token

print("Loading SFT model...")
sft_model = AutoModelForCausalLM.from_pretrained(
    config.sft_model_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)
sft_model.eval()

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
base_model = AutoModelForCausalLM.from_pretrained(
    config.reward_model_dir,
    torch_dtype=torch.float16,
    device_map="auto"
)

reward_model = RewardModel(base_model)
checkpoint = torch.load(os.path.join(config.reward_model_dir, "reward_model.pt"))
reward_model.reward_head.load_state_dict(checkpoint['reward_head_state_dict'])

# FIX: Convert reward head to float16 to match base model
reward_model.reward_head = reward_model.reward_head.to(torch.float16)
reward_model = reward_model.to(device)
reward_model.eval()

print("Model dtype check:")
print(f"Base model dtype: {next(base_model.parameters()).dtype}")
print(f"Reward head dtype: {next(reward_model.reward_head.parameters()).dtype}")

# Load prompts
print("\nLoading prompts...")
dataset = load_dataset(config.dataset_name, split="train")
dataset = dataset.select(range(min(config.num_gen_samples, len(dataset))))

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

for prompt in tqdm(prompts, desc="Generating responses"):
    responses = []

    # Generate multiple responses
    for _ in range(config.num_responses_per_prompt):
        input_text = f"Human: {prompt}\n\nAssistant:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=config.max_length).to(device)

        with torch.no_grad():
            outputs = sft_model.generate(
                **inputs,
                max_new_tokens=150,
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

print(f"\nGenerated {len(synthetic_data)} synthetic preference pairs")

# Save synthetic data
with open(config.synthetic_data_path, 'w') as f:
    json.dump(synthetic_data, f, indent=2)

print(f"✓ Synthetic preferences saved to {config.synthetic_data_path}")
print("\n✓ Synthetic Data Generation Complete!")

# Print statistics
if synthetic_data:
    chosen_scores = [d['chosen_score'] for d in synthetic_data]
    rejected_scores = [d['rejected_score'] for d in synthetic_data]
    score_diffs = [d['chosen_score'] - d['rejected_score'] for d in synthetic_data]

    print("\n=== Statistics ===")
    print(f"Average chosen score: {np.mean(chosen_scores):.4f}")
    print(f"Average rejected score: {np.mean(rejected_scores):.4f}")
    print(f"Average score difference: {np.mean(score_diffs):.4f}")
    print(f"Min score difference: {np.min(score_diffs):.4f}")
    print(f"Max score difference: {np.max(score_diffs):.4f}")