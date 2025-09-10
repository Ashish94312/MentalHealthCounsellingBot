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

# ----------------------------
# Device info
# ----------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ----------------------------
# Dataset
# ----------------------------
print("📥 Loading dataset...")
dataset = load_dataset("Amod/mental_health_counseling_conversations")


if "validation" not in dataset:
    dataset = dataset["train"].train_test_split(test_size=0.1)
else:
    dataset = dataset

# ----------------------------
# Tokenizer
# ----------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinylama-mental-health"
OFFLOAD_DIR = "./offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 256

# ----------------------------
# Load model on MPS only (no CPU offload)
# ----------------------------
print("📥 Loading model on MPS only...")
if device != "mps":
    raise RuntimeError("MPS is not available. Enable MPS or switch device.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": "mps"},   # force all modules to MPS
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Enable memory-efficient attention if supported
try:
    setattr(model.config, "attn_implementation", "sdpa")
except Exception:
    pass

# ----------------------------
# LoRA / PEFT
# ----------------------------
print("🔧 Setting up LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Print trainable parameters manually
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

# Gradient checkpointing & disable cache
model.gradient_checkpointing_enable()
model.config.use_cache = False
# Ensure inputs require grads when gradient checkpointing is enabled
try:
    model.enable_input_require_grads()
except AttributeError:
    pass

# ----------------------------
# Data preprocessing
# ----------------------------
def format_conversation(example):
    context = example.get("Context", "")
    response = example.get("Response", "")
    return {"text": f"Client: {context}\nCounselor: {response}".strip()}

formatted_dataset = dataset.map(format_conversation)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=400
    )

tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

# ----------------------------
# Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,  # small batch for MPS memory
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=False,                    # MUST be False on MPS
    bf16=False,
    gradient_checkpointing=True,
    optim="adamw_torch",          # avoid bitsandbytes optimizers on CPU/MPS
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    report_to=[],
    dataloader_pin_memory=False,
)

# ----------------------------
# Data collator
# ----------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ----------------------------
# Trainer
# ----------------------------

class NoOpMoveTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        return model

trainer = NoOpMoveTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset.get("validation"),
    data_collator=data_collator
)

# ----------------------------
# Train & Save
# ----------------------------
print("🏋️ Starting training...")
trainer.train()

print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete! Model saved at", OUTPUT_DIR)
