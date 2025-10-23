

"""
Data loading and processing utilities
"""

import json
from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path

def load_hh_rlhf(num_samples=5000, seed=42):
    """Load and sample HH-RLHF dataset"""
    print(f"📥 Loading HH-RLHF dataset...")
    
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    print(f"Total samples: {len(dataset)}")
    
    if len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
        print(f"Sampled {num_samples} examples")
    
    return dataset

def process_hh_rlhf_sample(example):
    """Extract prompt and responses from HH-RLHF format"""
    chosen_text = example['chosen']
    rejected_text = example['rejected']
    
    # Extract prompt and responses
    if "Assistant:" in chosen_text:
        parts = chosen_text.split("Assistant:")
        prompt = parts[0].replace("Human:", "").strip()
        chosen_response = parts[1].strip() if len(parts) > 1 else ""
    else:
        prompt = chosen_text[:100]
        chosen_response = chosen_text
    
    if "Assistant:" in rejected_text:
        rejected_response = rejected_text.split("Assistant:")[1].strip()
    else:
        rejected_response = rejected_text
    
    return {
        'prompt': prompt,
        'chosen': chosen_response,
        'rejected': rejected_response
    }

def create_train_test_split(dataset, train_split=0.9, seed=42):
    """Split dataset into train and test"""
    split_idx = int(len(dataset) * train_split)
    
    dataset = dataset.shuffle(seed=seed)
    train_dataset = dataset.select(range(split_idx))
    test_dataset = dataset.select(range(split_idx, len(dataset)))
    
    return train_dataset, test_dataset

def save_processed_data(train_dataset, test_dataset, output_dir):
    """Save processed data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save as JSON
    train_data = [dict(ex) for ex in train_dataset]
    test_data = [dict(ex) for ex in test_dataset]
    
    with open(output_dir / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_dir / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    # Save as HF dataset
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    dataset_dict.save_to_disk(str(output_dir / "hh_rlhf_processed"))
    
    print(f" Data saved to {output_dir}")

def load_processed_data(data_dir):
    """Load processed data"""
    data_dir = Path(data_dir)
    dataset = DatasetDict.load_from_disk(str(data_dir / "hh_rlhf_processed"))
    return dataset['train'], dataset['test']