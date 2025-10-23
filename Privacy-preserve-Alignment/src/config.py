

"""
Configuration file for Project 4
"""

from pathlib import Path

class Config:
    """Global configuration"""
    
    # Paths
    BASE_DIR = Path("/kaggle/working")
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    
    # Model names
    POLICY_MODEL = "gpt2"  # GPT-2 Small (124M)
    REWARD_MODEL = "distilbert-base-uncased"  # DistilBERT (66M)
    
    # Dataset
    DATASET_NAME = "Anthropic/hh-rlhf"
    NUM_SAMPLES = 5000
    TRAIN_SPLIT = 0.9
    SEED = 42
    
    # Training hyperparameters
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 5e-5
    NUM_EPOCHS = 3
    
    # LoRA hyperparameters
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["c_attn", "c_proj"]
    
    # Privacy hyperparameters
    EPSILON_VALUES = [1.0, 4.0, 8.0, 16.0]
    DELTA = 1e-5
    MAX_GRAD_NORM = 1.0
    
    # Evaluation
    EVAL_BATCH_SIZE = 8
    NUM_EVAL_SAMPLES = 500
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        for dir_path in [cls.DATA_DIR, cls.MODEL_DIR, cls.RESULTS_DIR, cls.CHECKPOINT_DIR]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            'policy_model': cls.POLICY_MODEL,
            'reward_model': cls.REWARD_MODEL,
            'dataset_name': cls.DATASET_NAME,
            'num_samples': cls.NUM_SAMPLES,
            'train_split': cls.TRAIN_SPLIT,
            'max_length': cls.MAX_LENGTH,
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'num_epochs': cls.NUM_EPOCHS,
            'lora_r': cls.LORA_R,
            'lora_alpha': cls.LORA_ALPHA,
            'epsilon_values': cls.EPSILON_VALUES,
            'delta': cls.DELTA,
            'max_grad_norm': cls.MAX_GRAD_NORM,
            'seed': cls.SEED,
        }