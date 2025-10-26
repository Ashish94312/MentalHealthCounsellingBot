#!/usr/bin/env python3
"""
Mental Health Chatbot Fine-tuning Script

This script fine-tunes TinyLlama for mental health conversations using LoRA.
I chose this approach after experimenting with full fine-tuning and finding
that LoRA provides better results with less computational overhead.

Key decisions made during development:
- r=32: Found this gave better response quality than r=16
- Gradient accumulation: Simulates larger batch sizes on limited hardware
- Chat template: Crucial for maintaining conversational flow
- 2 epochs: Sweet spot between underfitting and overfitting
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


def select_device() -> str:
    """
    Device selection with preference for Apple Silicon MPS.
    I prioritized MPS over CUDA because I developed this on a MacBook Air M1,
    and MPS provides excellent performance for this model size.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_data() -> dict:
    """
    Load and split the mental health dataset.
    ShenLab/MentalChat16K was the sweet spot - good quality, appropriate content.
    """
    print("📥 Loading ShenLab/MentalChat16K dataset...")
    
    dataset = load_dataset("ShenLab/MentalChat16K")
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset["validation"] = dataset["test"]
    del dataset["test"]
    
    return dataset


def build_tokenizer(model_name: str, max_length: int) -> AutoTokenizer:
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length
    return tokenizer


def build_model(model_name: str, device: str) -> AutoModelForCausalLM:
    print(f"📥 Loading base model on {device}...")
    if device == "mps":
        torch_dtype = torch.float16
        device_map = {"": "mps"}
    elif device == "cuda":
        torch_dtype = torch.float16
        device_map = {"": 0}
    else:
        torch_dtype = torch.float32
        device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )

    try:
        setattr(model.config, "attn_implementation", "sdpa")
    except Exception:
        pass

    return model


def apply_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """
    Apply LoRA configuration to the model.
    I experimented with different r values and found r=32 provides
    the best balance between adaptation capability and parameter efficiency.
    The target modules cover both attention and MLP layers for comprehensive adaptation.
    """
    print("🔧 Setting up LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    return get_peft_model(model, lora_config)


def format_conversation_factory(tokenizer: AutoTokenizer):
    def format_conversation(example):
        instruction = example.get("instruction", "You are a supportive and professional mental health counselor.")
        user_input = example.get("input", "")
        output = example.get("output", "")

        text = (
            f"<|system|>\n{instruction}\n"
            f"<|user|>\n{user_input}\n"
            f"<|assistant|>\n{output}{tokenizer.eos_token}"
        )
        return {"text": text}

    return format_conversation


def build_tokenized_dataset(dataset: dict, tokenizer: AutoTokenizer, max_length: int):
    print("🧹 Formatting and tokenizing dataset...")
    formatted = dataset.map(format_conversation_factory(tokenizer))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    return formatted.map(tokenize_function, batched=True)


def build_trainer(model, tokenized_dataset, tokenizer, output_dir: str, device: str, max_length: int) -> Trainer:
    print("⚙️  Preparing trainer...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        eval_strategy="steps",
        eval_steps=500,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        fp16=(device == "cuda"),
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        report_to=[],
        dataloader_pin_memory=False,
        seed=42,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    class NoOpMoveTrainer(Trainer):
        def _move_model_to_device(self, model, device):
            return model

    return NoOpMoveTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        data_collator=data_collator
    )


def main():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "./tinylama-mental-health-mentalchat16k-v4"
    max_length = 1024

    device = select_device()
    print(f"Using device: {device}")

    dataset = load_data()
    tokenizer = build_tokenizer(model_name, max_length)
    base_model = build_model(model_name, device)
    peft_model = apply_lora(base_model)

    peft_model.gradient_checkpointing_enable()
    peft_model.config.use_cache = False
    try:
        peft_model.enable_input_require_grads()
    except AttributeError:
        pass

    tokenized_dataset = build_tokenized_dataset(dataset, tokenizer, max_length)
    trainer = build_trainer(peft_model, tokenized_dataset, tokenizer, output_dir, device, max_length)

    print("🏋️ Starting training...")
    trainer.train()

    print("💾 Saving model...")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Training complete! Model saved at", output_dir)


if __name__ == "__main__":
    main()


