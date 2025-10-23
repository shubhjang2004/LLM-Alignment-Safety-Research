

"""
Evaluation utilities
"""

import torch
import numpy as np
from tqdm.auto import tqdm

def generate_responses(model, tokenizer, prompts, max_new_tokens=128, batch_size=1):
    """
    Generate responses for prompts
    
    Args:
        model: Causal LM model
        tokenizer: Tokenizer
        prompts: List of prompts
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for generation
        
    Returns:
        List of generated responses
    """
    model.eval()
    responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Remove prompt from response
        for prompt, response in zip(batch_prompts, batch_responses):
            response_only = response[len(prompt):].strip()
            responses.append(response_only)
    
    return responses

def compute_reward_scores(prompts, responses, reward_model, reward_tokenizer, batch_size=8):
    """
    Compute reward scores for prompt-response pairs
    
    Args:
        prompts: List of prompts
        responses: List of responses
        reward_model: Reward model
        reward_tokenizer: Reward tokenizer
        batch_size: Batch size
        
    Returns:
        Array of reward scores
    """
    reward_model.eval()
    rewards = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Computing rewards"):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]
        
        # Combine prompt and response
        texts = [f"{p} {r}" for p, r in zip(batch_prompts, batch_responses)]
        
        inputs = reward_tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(reward_model.device)
        
        with torch.no_grad():
            outputs = reward_model(**inputs)
            batch_rewards = outputs.logits.squeeze(-1)
        
        rewards.extend(batch_rewards.cpu().numpy())
    
    return np.array(rewards)

def evaluate_model(model, tokenizer, test_prompts, reward_model, reward_tokenizer, num_samples=200):
    """
    Comprehensive model evaluation
    
    Returns:
        Dictionary of metrics
    """
    # Sample prompts
    eval_prompts = test_prompts[:num_samples]
    
    # Generate responses
    responses = generate_responses(model, tokenizer, eval_prompts)
    
    # Compute rewards
    rewards = compute_reward_scores(eval_prompts, responses, reward_model, reward_tokenizer)
    
    metrics = {
        'mean_reward': float(rewards.mean()),
        'std_reward': float(rewards.std()),
        'median_reward': float(np.median(rewards)),
        'min_reward': float(rewards.min()),
        'max_reward': float(rewards.max()),
        'num_samples': len(rewards)
    }
    
    return metrics, rewards, responses