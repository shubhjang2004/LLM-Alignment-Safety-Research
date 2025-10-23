



# Verify GPU
import torch
print(f" PyTorch version: {torch.__version__}")
print(f" CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ═══════════════════════════════════════════════════════════════════════════
#  Import Libraries
# ═══════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
from datasets import load_dataset
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TaskConfig:
    """Configuration for each task"""
    name: str
    dataset_name: str
    dataset_config: Optional[str]
    prompt_template: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    num_samples: int = 5000

# Define 4 specialized tasks
TASKS = {
    "medical": TaskConfig(
        name="medical_conversation",
        dataset_name="medalpaca/medical_meadow_medical_flashcards",
        dataset_config=None,
        prompt_template="### Medical Question:\n{input}\n\n### Answer:\n{output}",
        lora_r=16,
        lora_alpha=32,
        num_samples=5000
    ),
    "code": TaskConfig(
        name="code_generation",
        dataset_name="iamtarun/python_code_instructions_18k_alpaca",
        dataset_config=None,
        prompt_template="### Instruction:\n{input}\n\n### Code:\n{output}",
        lora_r=16,
        lora_alpha=32,
        num_samples=5000
    ),
    "math": TaskConfig(
        name="math_reasoning",
        dataset_name="gsm8k",
        dataset_config="main",
        prompt_template="### Problem:\n{input}\n\n### Solution:\n{output}",
        lora_r=16,
        lora_alpha=32,
        num_samples=5000
    ),
    "creative": TaskConfig(
        name="creative_writing",
        dataset_name="euclaise/writingprompts",
        dataset_config=None,
        prompt_template="### Writing Prompt:\n{input}\n\n### Story:\n{output}",
        lora_r=16,
        lora_alpha=32,
        num_samples=3000  # Smaller for creative to avoid very long texts
    )
}

# Model and training configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH = 512
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 100

print("✓ Configuration loaded")
print(f"  Model: {MODEL_NAME}")
print(f"  Tasks: {list(TASKS.keys())}")
print(f"  Max Length: {MAX_LENGTH}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")

# ═══════════════════════════════════════════════════════════════════════════
#  Dataset Class
# ═══════════════════════════════════════════════════════════════════════════

class TaskDataset(Dataset):
    """Dataset for a specific task"""
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

print("✓ Dataset class defined")

# ═══════════════════════════════════════════════════════════════════════════
#  Data Preparation Functions
# ═══════════════════════════════════════════════════════════════════════════

def prepare_medical_data(dataset, config: TaskConfig):
    """Prepare medical conversation data"""
    print(f"  Preparing medical data (target: {config.num_samples} samples)...")
    data = []
    for i, item in enumerate(dataset):
        if len(data) >= config.num_samples:
            break
        
        input_text = item.get('input', '') or item.get('instruction', '')
        output_text = item.get('output', '') or item.get('response', '')
        
        if input_text and output_text:
            text = config.prompt_template.format(input=input_text, output=output_text)
            data.append({'text': text})
    
    print(f"   Prepared {len(data)} medical samples")
    return data

def prepare_code_data(dataset, config: TaskConfig):
    """Prepare code generation data"""
    print(f"  Preparing code data (target: {config.num_samples} samples)...")
    data = []
    for i, item in enumerate(dataset):
        if len(data) >= config.num_samples:
            break
        
        input_text = item.get('instruction', '')
        output_text = item.get('output', '')
        
        if input_text and output_text:
            text = config.prompt_template.format(input=input_text, output=output_text)
            data.append({'text': text})
    
    print(f"  Prepared {len(data)} code samples")
    return data

def prepare_math_data(dataset, config: TaskConfig):
    """Prepare math reasoning data"""
    print(f"  Preparing math data (target: {config.num_samples} samples)...")
    data = []
    for i, item in enumerate(dataset):
        if len(data) >= config.num_samples:
            break
        
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        if question and answer:
            text = config.prompt_template.format(input=question, output=answer)
            data.append({'text': text})
    
    print(f"   Prepared {len(data)} math samples")
    return data

def prepare_creative_data(dataset, config: TaskConfig):
    """Prepare creative writing data"""
    print(f"  Preparing creative data (target: {config.num_samples} samples)...")
    data = []
    for i, item in enumerate(dataset):
        if len(data) >= config.num_samples:
            break
        
        prompt = item.get('prompt', '')
        story = item.get('story', '') or item.get('text', '')
        
        if prompt and story and len(story) > 100:
            # Truncate very long stories
            story = story[:1000]
            text = config.prompt_template.format(input=prompt, output=story)
            data.append({'text': text})
    
    print(f"   Prepared {len(data)} creative samples")
    return data

print(" Data preparation functions defined")

# ═══════════════════════════════════════════════════════════════════════════
#  Load Tokenizer and Base Model
# ═══════════════════════════════════════════════════════════════════════════

print("Loading tokenizer and base model...")
print(f"Model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print(" Tokenizer loaded")
print(f"  Vocab size: {len(tokenizer)}")
print(f"  Pad token: {tokenizer.pad_token}")

# Note: We'll load the base model fresh for each task to avoid memory issues
print(" Ready to load base model (will load per task)")

# ═══════════════════════════════════════════════════════════════════════════
#  Load All Task Datasets
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("LOADING ALL TASK DATASETS")
print("="*70)

task_datasets = {}

# Medical data
print("\n[1/4] Loading Medical Dataset...")
try:
    ds = load_dataset(TASKS['medical'].dataset_name, split='train')
    data = prepare_medical_data(ds, TASKS['medical'])
    task_datasets['medical'] = TaskDataset(data, tokenizer, max_length=MAX_LENGTH)
    print(f" Medical dataset ready: {len(data)} samples")
except Exception as e:
    print(f" Medical data failed: {e}")

# Code data
print("\n[2/4] Loading Code Dataset...")
try:
    ds = load_dataset(TASKS['code'].dataset_name, split='train')
    data = prepare_code_data(ds, TASKS['code'])
    task_datasets['code'] = TaskDataset(data, tokenizer, max_length=MAX_LENGTH)
    print(f" Code dataset ready: {len(data)} samples")
except Exception as e:
    print(f" Code data failed: {e}")

# Math data
print("\n[3/4] Loading Math Dataset...")
try:
    ds = load_dataset(TASKS['math'].dataset_name, TASKS['math'].dataset_config, split='train')
    data = prepare_math_data(ds, TASKS['math'])
    task_datasets['math'] = TaskDataset(data, tokenizer, max_length=MAX_LENGTH)
    print(f"Math dataset ready: {len(data)} samples")
except Exception as e:
    print(f" Math data failed: {e}")

# Creative writing data
print("\n[4/4] Loading Creative Writing Dataset...")
try:
    ds = load_dataset(TASKS['creative'].dataset_name, split='train')
    data = prepare_creative_data(ds, TASKS['creative'])
    task_datasets['creative'] = TaskDataset(data, tokenizer, max_length=MAX_LENGTH)
    print(f" Creative dataset ready: {len(data)} samples")
except Exception as e:
    print(f" Creative data failed: {e}")

print("\n" + "="*70)
print(f" LOADED {len(task_datasets)}/4 DATASETS SUCCESSFULLY")
print("="*70)

# Verify datasets
for task_name, dataset in task_datasets.items():
    print(f"  {task_name}: {len(dataset)} samples")

# ═══════════════════════════════════════════════════════════════════════════
#  Define Training Function
# ═══════════════════════════════════════════════════════════════════════════

def train_lora_expert(task_name: str, task_config: TaskConfig, tokenizer, dataset):
    """Train a single LoRA expert for a specific task"""
    print("\n" + "="*70)
    print(f"TRAINING LORA EXPERT: {task_name.upper()}")
    print("="*70)
    
    # Load fresh base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    print(" Base model loaded")
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=task_config.lora_r,
        lora_alpha=task_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=task_config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Create PEFT model
    model = get_peft_model(base_model, lora_config)
    print(" LoRA adapter attached")
    model.print_trainable_parameters()
    
    # Training arguments
    output_dir = f"./lora_experts/{task_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save LoRA weights
    print("\nSaving LoRA adapter...")
    model.save_pretrained(output_dir)
    print(f" LoRA expert saved to: {output_dir}")
    
    # Clean up
    del model
    del base_model
    torch.cuda.empty_cache()
    
    return output_dir

print(" Training function defined")

# ═══════════════════════════════════════════════════════════════════════════
# Train Medical Expert
# ═══════════════════════════════════════════════════════════════════════════

if 'medical' in task_datasets:
    medical_path = train_lora_expert(
        'medical',
        TASKS['medical'],
        tokenizer,
        task_datasets['medical']
    )
    print(f"\n MEDICAL EXPERT TRAINING COMPLETE ✓✓✓")
else:
    print(" Medical dataset not available, skipping...")
    medical_path = None

# ═══════════════════════════════════════════════════════════════════════════
#  Train Code Expert
# ═══════════════════════════════════════════════════════════════════════════

if 'code' in task_datasets:
    code_path = train_lora_expert(
        'code',
        TASKS['code'],
        tokenizer,
        task_datasets['code']
    )
    print(f"\n CODE EXPERT TRAINING COMPLETE ✓✓✓")
else:
    print("Code dataset not available, skipping...")
    code_path = None

# ═══════════════════════════════════════════════════════════════════════════
#  Train Math Expert
# ═══════════════════════════════════════════════════════════════════════════

if 'math' in task_datasets:
    math_path = train_lora_expert(
        'math',
        TASKS['math'],
        tokenizer,
        task_datasets['math']
    )
    print(f"\nMATH EXPERT TRAINING COMPLETE ✓✓✓")
else:
    print("Math dataset not available, skipping...")
    math_path = None

# ═══════════════════════════════════════════════════════════════════════════
#  Train Creative Writing Expert
# ═══════════════════════════════════════════════════════════════════════════

if 'creative' in task_datasets:
    creative_path = train_lora_expert(
        'creative',
        TASKS['creative'],
        tokenizer,
        task_datasets['creative']
    )
    print(f"\n✓✓✓ CREATIVE WRITING EXPERT TRAINING COMPLETE ✓✓✓")
else:
    print("⚠ Creative dataset not available, skipping...")
    creative_path = None

# ═══════════════════════════════════════════════════════════════════════════
# Save Configuration & Summary
# ═══════════════════════════════════════════════════════════════════════════

# Collect trained paths
lora_paths = {}
if medical_path:
    lora_paths['medical'] = medical_path
if code_path:
    lora_paths['code'] = code_path
if math_path:
    lora_paths['math'] = math_path
if creative_path:
    lora_paths['creative'] = creative_path

# Save configuration
config = {
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "tasks": {k: {
        "name": v.name,
        "lora_r": v.lora_r,
        "lora_alpha": v.lora_alpha,
        "lora_dropout": v.lora_dropout,
        "prompt_template": v.prompt_template
    } for k, v in TASKS.items()},
    "lora_paths": lora_paths,
    "training_config": {
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS
    }
}

with open("./experts_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "="*70)
print(" ALL LORA EXPERTS TRAINING COMPLETE! ")
print("="*70)
print("\n Training Summary:")
print(f"  Experts trained: {len(lora_paths)}/4")
print(f"   Base model: {MODEL_NAME}")
print(f"   LoRA rank: {TASKS['medical'].lora_r}")
print(f"   Training epochs: {NUM_EPOCHS}")
print("\n Saved files:")
print(f"   LoRA experts: ./lora_experts/")
for task, path in lora_paths.items():
    print(f"    - {task}: {path}")
print(f"   Configuration: ./experts_config.json")
print("\n Next Step: Run Notebook 2 to train the router!")
print("="*70)




# ═══════════════════════════════════════════════════════════════════════════
#  Quick Test (Optional)
# ═══════════════════════════════════════════════════════════════════════════

# Quick test of a trained expert
def quick_test_expert(task_name: str, test_prompt: str):
    """Quick test of a trained expert"""
    print(f"\n{'='*70}")
    print(f"Testing {task_name.upper()} Expert")
    print(f"{'='*70}")
    
    from peft import PeftModel
    
    # Load base model
    print("Loading models...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, f"./lora_experts/{task_name}")
    model.eval()
    
    # Generate
    print(f"\n Prompt: {test_prompt}")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n Response:\n{response}")
    
    # Cleanup
    del model
    del base_model
    torch.cuda.empty_cache()

# Uncomment to test an expert:
# quick_test_expert('medical', '### Medical Question:\nWhat are the symptoms of diabetes?\n\n### Answer:\n')
# quick_test_expert('code', '### Instruction:\nWrite a function to reverse a string\n\n### Code:\n')

