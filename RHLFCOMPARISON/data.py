

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import pandas as pd
from typing import List, Dict
import json

print("="*60)
print("RLHF Data Preparation - Compact Dataset Creation")
print("="*60)

# Configuration
MAX_TOKEN_LENGTH = 250
MIN_TOKEN_LENGTH = 20
TARGET_SAMPLES = 5000  # Smaller, high-quality dataset
VALIDATION_SPLIT = 0.15

# Initialize tokenizer for length checking
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def count_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(tokenizer.encode(text, add_special_tokens=True))

def filter_by_length(example: Dict) -> bool:
    """Filter examples by token length"""
    prompt_len = count_tokens(example['prompt'])
    chosen_len = count_tokens(example['chosen'])
    rejected_len = count_tokens(example['rejected'])
    
    # Check that full sequences fit within limit
    chosen_total = prompt_len + chosen_len
    rejected_total = prompt_len + rejected_len
    
    return (chosen_total <= MAX_TOKEN_LENGTH and 
            rejected_total <= MAX_TOKEN_LENGTH and
            prompt_len >= MIN_TOKEN_LENGTH)

def prepare_hh_rlhf_compact():
    """Prepare compact version of HH-RLHF dataset"""
    print("\n Loading Anthropic HH-RLHF dataset...")
    
    # Load only the helpful subset (smaller and cleaner)
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
    
    print(f"Original dataset size: {len(dataset)}")
    
    # Process and filter
    processed_data = []
    
    for idx, example in enumerate(dataset):
        if len(processed_data) >= TARGET_SAMPLES:
            break
            
        # HH-RLHF format: chosen and rejected are full conversations
        chosen_text = example['chosen']
        rejected_text = example['rejected']
        
        # Extract the last turn (most relevant for comparison)
        # Format: "\n\nHuman: ... \n\nAssistant: ..."
        try:
            # Get the prompt (everything before last Assistant response)
            if "\n\nAssistant:" in chosen_text:
                parts = chosen_text.split("\n\nAssistant:")
                prompt = parts[0] + "\n\nAssistant:"
                chosen_response = parts[-1].strip()
                
                rejected_parts = rejected_text.split("\n\nAssistant:")
                rejected_response = rejected_parts[-1].strip()
                
                # Check length
                temp_example = {
                    'prompt': prompt,
                    'chosen': chosen_response,
                    'rejected': rejected_response
                }
                
                if filter_by_length(temp_example):
                    processed_data.append(temp_example)
                    
        except Exception as e:
            continue
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} examples, kept {len(processed_data)}...")
    
    print(f"\n Filtered to {len(processed_data)} examples within token limit")
    
    return processed_data

def prepare_summary_dataset():
    """Alternative: Use Reddit TL;DR summarization comparisons"""
    print("\n summarization comparison dataset...")
    
    # This is a backup option with naturally shorter texts
    dataset = load_dataset("openai/summarize_from_feedback", "comparisons", split="train")
    
    processed_data = []
    
    for idx, example in enumerate(dataset):
        if len(processed_data) >= TARGET_SAMPLES:
            break
        
        # This dataset has post (prompt) and summaries
        info = example['info']
        prompt = info['post']
        
        # Get the two summaries being compared
        summaries = example['summaries']
        choice = example['choice']
        
        if len(summaries) >= 2 and choice in [0, 1]:
            temp_example = {
                'prompt': f"Summarize the following post:\n\n{prompt}\n\nSummary:",
                'chosen': summaries[choice]['text'],
                'rejected': summaries[1 - choice]['text']
            }
            
            if filter_by_length(temp_example):
                processed_data.append(temp_example)
        
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1} examples, kept {len(processed_data)}...")
    
    print(f"\n Filtered to {len(processed_data)} examples within token limit")
    
    return processed_data

def create_dataset_splits(data: List[Dict]) -> DatasetDict:
    """Create train/validation splits"""
    print("\n Creating train/validation splits...")
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(data)
    
    # Split
    val_size = int(len(data) * VALIDATION_SPLIT)
    train_data = data[val_size:]
    val_data = data[:val_size]
    
    # Convert to Dataset objects
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    
    return dataset_dict

def analyze_dataset(dataset_dict: DatasetDict):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for split_name, split_data in dataset_dict.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Samples: {len(split_data)}")
        
        # Token length statistics
        prompt_lengths = []
        chosen_lengths = []
        rejected_lengths = []
        
        for example in split_data:
            prompt_lengths.append(count_tokens(example['prompt']))
            chosen_lengths.append(count_tokens(example['chosen']))
            rejected_lengths.append(count_tokens(example['rejected']))
        
        print(f"  Prompt tokens - Mean: {sum(prompt_lengths)/len(prompt_lengths):.1f}, "
              f"Max: {max(prompt_lengths)}")
        print(f"  Chosen tokens - Mean: {sum(chosen_lengths)/len(chosen_lengths):.1f}, "
              f"Max: {max(chosen_lengths)}")
        print(f"  Rejected tokens - Mean: {sum(rejected_lengths)/len(rejected_lengths):.1f}, "
              f"Max: {max(rejected_lengths)}")
        
        total_lengths = [p + c for p, c in zip(prompt_lengths, chosen_lengths)]
        print(f"  Total sequence - Mean: {sum(total_lengths)/len(total_lengths):.1f}, "
              f"Max: {max(total_lengths)}")

def save_dataset(dataset_dict: DatasetDict, path: str = "./rlhf_compact_data"):
    """Save processed dataset"""
    print(f"\n Saving dataset to {path}...")
    dataset_dict.save_to_disk(path)
    print(" Dataset saved!")
    
    # Also save as JSON for inspection
    for split_name, split_data in dataset_dict.items():
        json_path = f"{path}/{split_name}.json"
        with open(json_path, 'w') as f:
            json.dump(split_data.to_list(), f, indent=2)
        print(f"  Saved {split_name} split to {json_path}")

def show_examples(dataset_dict: DatasetDict, n: int = 3):
    """Display sample examples"""
    print("\n" + "="*60)
    print(f"SAMPLE EXAMPLES (first {n} from train)")
    print("="*60)
    
    for i, example in enumerate(dataset_dict['train'].select(range(n))):
        print(f"\n--- Example {i+1} ---")
        print(f"PROMPT ({count_tokens(example['prompt'])} tokens):")
        print(example['prompt'][:200] + "..." if len(example['prompt']) > 200 else example['prompt'])
        print(f"\nCHOSEN ({count_tokens(example['chosen'])} tokens):")
        print(example['chosen'][:150] + "..." if len(example['chosen']) > 150 else example['chosen'])
        print(f"\nREJECTED ({count_tokens(example['rejected'])} tokens):")
        print(example['rejected'][:150] + "..." if len(example['rejected']) > 150 else example['rejected'])

# Main execution
if __name__ == "__main__":
    print("\n Starting data preparation pipeline...\n")
    
    # Try HH-RLHF first (recommended)
    try:
        processed_data = prepare_hh_rlhf_compact()
    except Exception as e:
        print(f"  HH-RLHF failed: {e}")
        print("Trying summarization dataset as backup...")
        processed_data = prepare_summary_dataset()
    
    # Create splits
    dataset_dict = create_dataset_splits(processed_data)
    
    # Analyze
    analyze_dataset(dataset_dict)
    
    # Show examples
    show_examples(dataset_dict)
    
    # Save
    save_dataset(dataset_dict)
    
    print("\n" + "="*60)
    print(" DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run Notebook 2 to train the DistilBERT reward model")
    print("2. Use the trained reward model in Notebook 3 for PPO training")