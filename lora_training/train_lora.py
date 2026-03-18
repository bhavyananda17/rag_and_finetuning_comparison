"""
LoRA Fine-Tuning Training Script

This script trains a language model using LoRA (Low-Rank Adaptation) on a QA dataset.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from lora_training.config import *
from lora_training.dataset import prepare_dataset


def setup_model_and_tokenizer(model_name: str):
    """
    Load the base model and tokenizer, optionally with quantization.
    """
    print(f"[*] Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    return model, tokenizer


def setup_lora(model):
    """
    Configure and apply LoRA to the model.
    """
    print("[*] Setting up LoRA configuration...")
    
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def train():
    """
    Main training function.
    """
    print("\n" + "="*50)
    print("    LoRA FINE-TUNING PIPELINE")
    print("="*50 + "\n")
    
    # Check if QA dataset exists
    if not os.path.exists(QA_DATASET_PATH):
        print(f"[!] QA dataset not found at {QA_DATASET_PATH}")
        print("[*] Run generate_qa_dataset.py first.")
        return
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(BASE_MODEL)
    
    # Apply LoRA
    model = setup_lora(model)
    
    # Prepare dataset
    print(f"[*] Loading dataset from {QA_DATASET_PATH}...")
    dataset = prepare_dataset(QA_DATASET_PATH)
    print(f"[+] Dataset loaded: {len(dataset)} samples")
    
    # Split into train/validation
    split_dataset = dataset.train_test_split(test_size=VALIDATION_SPLIT, seed=42)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        max_steps=-1,
        logging_steps=10,
        save_steps=50,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=50,
        fp16=MIXED_PRECISION == "fp16",
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=None,  # Use default collator
    )
    
    # Train
    print("[*] Starting training...")
    trainer.train()
    
    # Save model
    print(f"[+] Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("[+] Training complete!")


if __name__ == "__main__":
    train()