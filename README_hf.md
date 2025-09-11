
# TinyLlama Mental Health (LoRA Adapter) — 16k checkpoint-3620

This repo hosts a LoRA adapter checkpoint for TinyLlama fine-tuned on mental health counseling style data.

- Base: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Domain: Mental health counseling
- Method: PEFT LoRA fine-tuning
- Context length: 16k
- Checkpoint: `checkpoint-3620`

## Usage (Apple Silicon, MPS only)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
adapter = 'sudojarvis/tinylama-mental-health-mentalchat16k-ckpt-3620'

device = torch.device('mps')  # user prefers MPS-only

# tokenizer
 tok = AutoTokenizer.from_pretrained(base)

# base model on MPS
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.to(device)

# load LoRA adapter
model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.float16)
model.to(device)

prompt = "I feel overwhelmed at work."
inputs = tok(prompt, return_tensors='pt').to(device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Files
- `adapter_model.safetensors`: LoRA weights
- `adapter_config.json`: PEFT configuration
- `tokenizer.*`, `special_tokens_map.json`, `chat_template.jinja`

## License
Apache-2.0
