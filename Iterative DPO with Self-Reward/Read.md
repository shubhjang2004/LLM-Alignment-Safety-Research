# Iterative Direct Preference Optimization (DPO) for LLM Alignment

A self-improving language model alignment system that uses iterative DPO training with synthetic preference generation, eliminating the need for external critics or continuous human feedback.

##  Project Overview

This project implements an **iterative Direct Preference Optimization (DPO)** pipeline that progressively improves language model alignment through self-generated synthetic preferences. Unlike traditional RLHF approaches that require expensive human feedback or external critic models, this system uses a trained reward model to generate its own training data across multiple rounds.

### Key Innovation
- **Self-Improving Loop**: Each DPO round generates better synthetic preferences from an already-improved model
- **No External Critics**: Uses a single trained reward model instead of API calls to external models (e.g., GPT-4, Claude)
- **Progressive Alignment**: Demonstrates measurable improvement across training iterations

##  Results Summary

| Metric | SFT Baseline | DPO Round 1 | DPO Round 2 | Improvement |
|--------|--------------|-------------|-------------|-------------|
| **Preference Accuracy** | - | 65.2% | 73.2% | +8.0% |
| **Reward Margin** | - | 0.692 | 0.769 | +11.1% |
| **Training Loss** | - | 0.345 | 0.386 | Stable |
| **Dataset Size** | 3,000 | 3,956 | 4,436 | +48% |

### Progressive Improvement
```
Round 0 (SFT) → Round 1 (DPO) → Round 2 (DPO)
    ↓                ↓                ↓
Baseline      +65% accuracy    +73% accuracy
              +0.69 margin     +0.77 margin
```

## 🏗️ Architecture

### Training Pipeline
```
┌─────────────────────────────────────────────────────────────┐
│                    ITERATIVE DPO PIPELINE                   │
└─────────────────────────────────────────────────────────────┘

1. Initial Training
   ├── Supervised Fine-Tuning (SFT)
   └── Reward Model Training
        ↓
2. Round 1: First DPO Iteration
   ├── Generate responses from SFT model (2 per prompt)
   ├── Score with reward model → Create synthetic preferences
   ├── Train DPO on original + synthetic data
   └── Result: DPO Round 1 model (65% accuracy, 0.69 margin)
        ↓
3. Round 2: Second DPO Iteration
   ├── Generate responses from DPO Round 1 model
   ├── Score with reward model → Better synthetic preferences
   ├── Train DPO on original + synthetic R1 + synthetic R2
   └── Result: DPO Round 2 model (73% accuracy, 0.77 margin)
```

### Model Architecture
- **Base Model**: GPT-2 (1.1B parameters)
- **Training Method**: LoRA (Low-Rank Adaptation)
  - Rank: 16
  - Alpha: 32
  - Trainable params: 4.5M (0.4% of total)
- **Precision**: FP16 (float16 for T4 GPU compatibility)

##  Quick Start

### Prerequisites
```bash
# Required libraries
pip install torch transformers datasets trl peft accelerate
pip install matplotlib seaborn tqdm
```

### Hardware Requirements
- **GPU**: NVIDIA T4 (15GB VRAM) or better
- **RAM**: 12GB+ system memory
- **Storage**: 10GB+ for models and data

### Training Steps

#### 1. Supervised Fine-Tuning (Optional - if starting from scratch)
```python
# Train base SFT model on HH-RLHF dataset
python train_sft.py
```

#### 2. Reward Model Training
```python
# Train reward model for scoring preferences
python train_reward_model.py
```

#### 3. Generate Synthetic Preferences (Round 1)
```python
# Generate 1000 preference pairs from SFT model
python generate_synthetic_round1.py
```

#### 4. DPO Training Round 1
```python
# Train first DPO iteration
python train_dpo_round1.py
# Time: ~1.5 hours on T4
```

#### 5. Generate Synthetic Preferences (Round 2)
```python
# Generate 500 preference pairs from DPO R1 model
python generate_synthetic_round2.py
```

#### 6. DPO Training Round 2
```python
# Train second DPO iteration
python train_dpo_round2.py
# Time: ~1.5 hours on T4
```

#### 7. Evaluation
```python
# Compare all models and generate visualizations
python evaluate_models.py
# Time: ~30 minutes
```

##  Project Structure
```
iterative-dpo/
├── README.md
├── requirements.txt
├── configs/
│   └── config.py                    # Hyperparameters and paths
├── scripts/
│   ├── train_sft.py                 # Supervised fine-tuning
│   ├── train_reward_model.py        # Reward model training
│   ├── generate_synthetic_round1.py # Synthetic data generation R1
│   ├── train_dpo_round1.py          # DPO training R1
│   ├── generate_synthetic_round2.py # Synthetic data generation R2
│   ├── train_dpo_round2.py          # DPO training R2
│   └── evaluate_models.py           # Model comparison
├── outputs/
│   ├── sft_model/                   # Base SFT model
│   ├── reward_model/                # Trained reward model
│   ├── dpo_round1/                  # DPO R1 LoRA adapters
│   ├── dpo_round2/                  # DPO R2 LoRA adapters
│   ├── synthetic_preferences.json   # Round 1 synthetic data
│   ├── synthetic_preferences_round2.json  # Round 2 synthetic data
│   ├── evaluation_results.json      # Evaluation metrics
│   └── plots/                       # Visualization outputs
└── notebooks/
    └── demo.ipynb                   # Colab-ready demo notebook
```

##  Key Metrics

### Training Metrics (DPO Round 2)
```
Training Progress:
├── Initial Loss: 0.554
├── Final Loss: 0.386 (-30%)
├── Reward Margin: 0.547 → 0.769 (+40%)
└── Preference Accuracy: 73.2%

Data Statistics:
├── Original HH-RLHF: 2,999 pairs
├── Synthetic Round 1: 956 pairs (95.6% success rate)
├── Synthetic Round 2: 481 pairs (96.2% success rate)
└── Total Training Data: 4,436 pairs
```

### Computational Efficiency

| Stage | Batch Size | Time (T4) | Memory |
|-------|------------|-----------|--------|
| Synthetic Gen R1 | N/A | ~60 min | 8GB |
| DPO Training R1 | 16 (effective) | ~90 min | 12GB |
| Synthetic Gen R2 | N/A | ~30 min | 8GB |
| DPO Training R2 | 16 (effective) | ~90 min | 12GB |
| **Total** | - | **~4.5 hours** | **12GB peak** |

##  Technical Details

### Reward Model Architecture
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model  # GPT-2 1.1B
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
```

**Loss Function**: Bradley-Terry preference loss
```python
loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()
```

### DPO Training Configuration
```python
DPOConfig(
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=3e-5,              # Lower for Round 2
    beta=0.1,                        # DPO temperature
    max_length=512,
    fp16=True,                       # T4 compatible
    gradient_checkpointing=True      # Memory optimization
)
```

### LoRA Configuration
```python
LoraConfig(
    r=16,                            # Rank
    lora_alpha=32,                   # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

##  Visualizations

The evaluation script generates three key plots:

1. **`reward_comparison.png`**: Bar chart comparing average rewards across models
2. **`reward_distributions.png`**: Histograms showing reward score distributions
3. **`improvement_trajectory.png`**: Line plot of cumulative improvement

## 🎓 Methodology

### Iterative DPO Algorithm
```
Algorithm: Iterative DPO with Synthetic Preferences
─────────────────────────────────────────────────────
Input: Dataset D, Base model M₀, Reward model R
Output: Aligned model M_n

1. Initialize: M₁ ← SFT(M₀, D)

2. For round i = 1 to n:
   
   a. Generate synthetic preferences:
      For each prompt p in D_new:
          - Generate responses: r₁, r₂ ~ M_i(p)
          - Score: s₁ = R(p, r₁), s₂ = R(p, r₂)
          - Create pair: (p, r_chosen, r_rejected) 
            where chosen = argmax(s₁, s₂)
   
   b. Combine datasets:
      D_train = D_original + D_synthetic_1 + ... + D_synthetic_i
   
   c. Train DPO:
      M_{i+1} ← DPO(M_i, D_train)
   
   d. Evaluate:
      Compute accuracy, reward margins
   
3. Return M_n
```

### Why It Works

1. **Quality Amplification**: Each round uses a better model to generate synthetic data
2. **Diverse Training Signal**: Multiple synthetic datasets provide varied learning signal
3. **No Distribution Shift**: Synthetic data comes from the same model family
4. **Scalable**: No human feedback required after initial reward model training

##  Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Round 1 | Round 2 | Notes |
|-----------|---------|---------|-------|
| Learning Rate | 5e-5 | 3e-5 | Lower for fine-tuning |
| Epochs | 3 | 2 | Faster convergence |
| Beta (DPO temp) | 0.1 | 0.1 | Controls KL penalty |
| Batch Size | 16 | 16 | Effective (1×16 accum) |
| LoRA Rank | 16 | 16 | Good efficiency/quality |
| Temperature (gen) | 0.9 | 0.85 | Slightly more focused |
| Samples Generated | 1000 | 500 | Quality over quantity |

### Memory Optimization Tips
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use FP16 training
training_args = DPOConfig(fp16=True)

# Sequential model loading
model_1 = load_model()
generate_responses()
del model_1
torch.cuda.empty_cache()

# Reduce batch size, increase accumulation
per_device_train_batch_size=1
gradient_accumulation_steps=16
```

##  Dataset

**Source**: [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
```python
from datasets import load_dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
```

**Format**: Conversational preference pairs
```json
{
  "chosen": "Human: [question]\n\nAssistant: [helpful response]",
  "rejected": "Human: [question]\n\nAssistant: [less helpful response]"
}
```

**Statistics**:
- Training samples: 160,800 pairs
- Test samples: 8,552 pairs
- Used in project: 3,000 training pairs (resource constraints)

##  Use Cases

This iterative DPO approach is suitable for:

-  **Low-resource alignment**: When human feedback is expensive/unavailable
-  **Domain adaptation**: Align models to specific domains using small seed datasets
-  **Continuous improvement**: Models that self-improve over deployment
-  **Research**: Studying emergent behaviors in iterative training
-  **Privacy-sensitive applications**: No external API calls required

##  Limitations & Future Work

### Current Limitations

1. **Helpfulness-Harmlessness Tradeoff**: Model may become more helpful but less cautious
2. **Reward Model Generalization**: Limited to distribution of training data
3. **Computational Cost**: Multiple training rounds required
4. **Scale**: Tested on 1.1B parameter model (larger models may behave differently)

### Future Improvements

- [ ] Implement Constitutional AI principles for safety constraints
- [ ] Add multi-objective optimization (helpfulness + harmlessness)
- [ ] Test on larger models (7B, 13B parameters)
- [ ] Implement online/continual learning
- [ ] Add human-in-the-loop validation
- [ ] Experiment with more DPO rounds (3-5 iterations)
- [ ] Balance synthetic data for safety

##  Contributing

Contributions are welcome! Areas for improvement:

1. **Safety mechanisms**: Add filtering for harmful synthetic preferences
2. **Evaluation metrics**: Implement GPT-4-as-judge or human evaluation
3. **Scalability**: Optimize for larger models
4. **Visualization**: Enhanced analysis tools

## References

1. **Direct Preference Optimization**: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)
2. **RLHF**: [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)
3. **LoRA**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
4. **Anthropic HH-RLHF**: [Bai et al., 2022](https://arxiv.org/abs/2204.05862)

##  License

MIT License - see LICENSE file for details

##  Acknowledgments

- Anthropic for the HH-RLHF dataset
- Hugging Face for Transformers and TRL libraries
- Google Colab for computational resources


** If you find this project useful, please consider starring the repository!**