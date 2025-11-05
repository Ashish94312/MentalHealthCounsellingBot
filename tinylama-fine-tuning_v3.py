#!/usr/bin/env python3

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
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("📥 Loading ShenLab/MentalChat16K dataset...")
dataset = load_dataset("ShenLab/MentalChat16K")

dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset["validation"] = dataset["test"]
del dataset["test"]

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinylama-mental-health-mentalchat16k"
OFFLOAD_DIR = "./offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 512

print("📥 Loading model on MPS only...")
if device != "mps":
    raise RuntimeError("MPS is not available. Enable MPS or switch device.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": "mps"},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

try:
    setattr(model.config, "attn_implementation", "sdpa")
except Exception:
    pass

print("🔧 Setting up LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_params} "
          f"|| trainable%: {100 * trainable_params / all_params:.2f}")

print_trainable_parameters(model)

model.gradient_checkpointing_enable()
model.config.use_cache = False
try:
    model.enable_input_require_grads()
except AttributeError:
    pass

def format_conversation(example):
    instruction = example.get("instruction", "")
    user_input = example.get("input", "")
    output = example.get("output", "")
    
    text = f"<|system|>\n{instruction}\n<|user|>\n{user_input}\n<|assistant|>\n{output}"
    return {"text": text}

formatted_dataset = dataset.map(format_conversation)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    eval_strategy="steps",
    eval_steps=500,
    learning_rate=1e-5,
    fp16=False,
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
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


class NoOpMoveTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        return model

trainer = NoOpMoveTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"] if "validation" in tokenized_dataset else None,
    data_collator=data_collator
)

print("🏋️ Starting training...")
trainer.train()

print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete! Model saved at", OUTPUT_DIR)
