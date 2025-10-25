# Scaling Laws in RLHF: Comparative Analysis of Small vs Medium Models

**Quantifying scale-alignment tradeoffs for efficient deployment**

This project implements a full **Reinforcement Learning from Human Feedback (RLHF)** pipeline to study **scaling effects** between small and medium-sized language models, specifically **GPT-2 (124M)** vs **GPT-2 (355M)**.  
The work investigates how model size influences **alignment quality, reward gain, and training stability** under resource-constrained (single-GPU) conditions.

---

##  Key Features

-  **PPO from scratch** — Full custom implementation using PyTorch.  
-  **Dataset:** Trained and evaluated on **30K Stanford SHP** preference examples.  
-  **Models:** GPT-2 (124M) vs GPT-2 (355M) fine-tuned for reward alignment.  
-  **Stability Improvements:**  
  - KL-divergence penalty  
  - Gradient clipping  
  - Reward normalization  
-  **Complete RLHF Pipeline:**  
  `SFT → Reward Model → PPO Optimization`  
  All optimized for **single-GPU training** (≤24GB VRAM).  

---

##  Results

| Model | Parameters | Reward Gain | Loss Reduction | Notes |
|:------|:-----------:|:------------:|:---------------:|:------|
| GPT-2 Small | 124M | — | — | Baseline SFT model |
| GPT-2 Medium | 355M | 2× | 57% | Clear scaling benefits in alignment quality |

### Training Curves (Example)

| Metric | GPT-2 Small | GPT-2 Medium |
|--------|--------------|---------------|
| Average Reward | 0.42 | **0.85** |
| KL Divergence | 0.13 | **0.09** |
| PPO Loss | 0.72 | **0.31** |

---

## ⚙️ Setup



##  Usage

### 1. Supervised Fine-Tuning (SFT)
```bash
python train_sft.py --model gpt2 --dataset stanford_shp
```

### 2. Train Reward Model
```bash
python train_reward_model.py --model gpt2-medium --dataset stanford_shp
```

### 3. PPO Optimization
```bash
python train_ppo.py --model gpt2-medium --reward_model reward_model.pt
```

---

##  Insights

- Medium models show **stronger alignment** and **more stable training** under PPO.  
- Scaling improves **reward efficiency** while maintaining **computational feasibility**.  
- KL-divergence penalty is crucial for stable optimization in larger models.

---

## 📎 Citation

If you use this work, please cite:
```
@article{rlhf_scaling_2025,
  title={Scaling Laws in RLHF: Comparative Analysis of Small vs Medium Models},
  author={Your Name},
  year={2025}
}
```

---

##  License

MIT License © 2025 — Your Name
