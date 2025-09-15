## Mental Health Counselling Bot

A Django-based web app that serves a fine-tuned TinyLlama conversational model for empathetic mental-health support. It provides a simple chat UI, crisis keyword detection with resource guidance, basic conversation memory, and an API endpoint for programmatic access. A training script is included to fine-tune TinyLlama on the MentalChat16K dataset using LoRA.

### Features
- **Empathetic assistant**: System prompt tuned for supportive guidance and boundaries
- **Crisis detection**: Keywords trigger immediate crisis-resource response
- **Identity handling**: Fast path for “who are you?” queries without model invocation
- **Conversation memory**: Short-term memory per `session_id` via Django cache
- **Web UI**: Minimal chat interface with typing indicator and crisis modal
- **API**: Single POST endpoint for chat replies
- **Training**: `tinylama-fine-tuning_v4.py` for LoRA fine-tuning on `ShenLab/MentalChat16K`

---

## Project structure
```
chatbot_project/
  chatbot/                 # Django app (views, urls, templates)
  chatbot_project/         # Django project (settings, urls, wsgi)

data/                      # Datasets and preprocessed artifacts
<model-dirs>/              # Fine-tuned adapters/tokenizer (e.g., tinylama-mental-health-mentalchat16k)
```
Key files:
- `chatbot_project/chatbot/views.py` – crisis detection, session memory, model loading/inference
- `chatbot_project/chatbot/templates/chatbot/chat.html` – chat UI
- `chatbot_project/chatbot/urls.py` and `chatbot_project/chatbot_project/urls.py` – routing
- `tinylama-fine-tuning_v4.py` – LoRA fine-tuning script on MentalChat16K
- `requirements.txt` – runtime and training dependencies

---

## Requirements
- Python 3.11+
- macOS with Apple Silicon works (supports MPS); CUDA GPU optional
- Disk space for model/tokenizer and adapter weights

Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Note: Training dependencies (e.g., `transformers`, `peft`, `datasets`, `accelerate`, `bitsandbytes`) are included.

---

## Configuration
Environment variables (optional but recommended):
- `SECRET_KEY` – Django secret key
- `DEBUG` – `true`/`false` (default: false)
- `ALLOWED_HOSTS` – comma-separated hosts (default: `*`); `testserver` is auto-added
- `CSRF_TRUSTED_ORIGINS` – comma-separated origins for CSRF
- `DATABASE_URL` – if set, overrides SQLite via `dj-database-url` (e.g., Postgres on Railway)

Static files are served by WhiteNoise in production. Database defaults to SQLite at `chatbot_project/chatbot_project/db.sqlite3` unless `DATABASE_URL` is provided.

---

## Running the web app (local)
Run migrations and start the server:
```bash
cd chatbot_project
python manage.py migrate
python manage.py runserver
```
Open the app at `http://127.0.0.1:8000/`.

### Initial model setup
By default the app loads a local fine-tuned directory:
- `tinylama-mental-health-mentalchat16k`

The path is set in `chatbot_project/chatbot/views.py` via `MODEL_PATH`. To use another adapter/model directory, change the value:
```python
MODEL_PATH = os.path.join(BASE_DIR, "tinylama-mental-health-mentalchat16k")
```
Ensure the directory contains tokenizer files and LoRA adapter weights compatible with `AutoModelForCausalLM` and `AutoTokenizer`.

### Performance notes
- Device auto-selection: MPS > CUDA > CPU
- On CPU, responses will be slower; consider reducing `max_new_tokens` in `generate` if needed
- For MPS/CUDA, fp16 is used automatically

---

## API usage
Endpoint: `POST /`

- Form-encoded or JSON body supported
- Fields:
  - `user_input` (string, required)
  - `session_id` (string, optional; generated if absent)

Example request (JSON):
```bash
curl -X POST http://127.0.0.1:8000/ \
  -H 'Content-Type: application/json' \
  -d '{"user_input": "I feel anxious about work.", "session_id": "demo-123"}'
```
Example success response:
```json
{
  "reply": "It sounds like work has been overwhelming...",
  "is_crisis": false,
  "session_id": "demo-123"
}
```
Crisis example response (keyword detected):
```json
{
  "reply": "I'm deeply concerned... National Suicide Prevention Lifeline: 988 ...",
  "is_crisis": true
}
```

---

## Safety and behavior
- Crisis keywords (e.g., “suicide”, “hurt myself”, “want to die”) trigger immediate crisis guidance and skip model generation.
- Identity queries (e.g., “who are you?”) return: “I’m an AI mental health support assistant…” via a fast path.
- Conversation memory: last 5 Q&A pairs (10 turns) are cached per `session_id` (24h expiry).

---

## Training (LoRA fine-tuning)
Script: `tinylama-fine-tuning_v4.py`

What it does:
- Loads `ShenLab/MentalChat16K`
- Formats to a chat-style template
- Applies LoRA to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Trains for a few epochs with gradient accumulation
- Saves adapter weights and tokenizer to `./tinylama-mental-health-mentalchat16k-v4`

Run:
```bash
python tinylama-fine-tuning_v4.py
```
Outputs:
- `tinylama-mental-health-mentalchat16k-v4/` with LoRA adapter and tokenizer files

Hardware guidance:
- macOS MPS or CUDA recommended
- On CPU, training is slow; consider reducing `max_length`, steps, or using a subset

Using the new adapter in the app:
1. Move/copy the trained directory into the repository root (or any path you prefer)
2. Update `MODEL_PATH` in `chatbot_project/chatbot/views.py` to point to it

---

## Deployment notes
- Project includes production-friendly settings: WhiteNoise static serving, security headers when `DEBUG=false`.
- `ALLOWED_HOSTS` and `CSRF_TRUSTED_ORIGINS` should be set for your domain.
- If deploying on a platform with env-provided Postgres, set `DATABASE_URL`.

Collect static (typically in CI/CD):
```bash
cd chatbot_project
python manage.py collectstatic --noinput
```
Run with gunicorn (example):
```bash
gunicorn chatbot_project.wsgi:application --bind 0.0.0.0:8000
```

---

## Troubleshooting
- "Model not found": Verify `MODEL_PATH` exists and contains `tokenizer.json` and adapter weights.
- Slow inference: Reduce `max_new_tokens` in `generate`, or run on MPS/CUDA.
- 403 CSRF in browser: Ensure you’re using form POSTs as in the built-in UI, or configure `CSRF_TRUSTED_ORIGINS` for your domain.
- Memory errors during training: Lower `max_length`, batch size, or enable gradient checkpointing (already enabled).


