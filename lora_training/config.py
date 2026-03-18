"""
Configuration settings for LoRA fine-tuning training.
"""

# Model Configuration
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
OUTPUT_DIR = "lora_adapter"

# Training Hyperparameters
LEARNING_RATE = 5e-4
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512

# LoRA Configuration
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Target attention layers

# Optimization
USE_GRADIENT_CHECKPOINTING = True
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# Hardware Configuration
DEVICE = "cuda"  # Change to "cpu" if needed
MIXED_PRECISION = "fp16"

# Dataset Configuration
QA_DATASET_PATH = "data/qa_dataset.json"
VALIDATION_SPLIT = 0.1