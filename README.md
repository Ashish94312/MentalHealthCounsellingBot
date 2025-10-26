# Mental Health Chatbot

A Django web app that runs a fine-tuned TinyLlama model for mental health support conversations.

## Features

- **Chat Interface**: Simple web chat interface
- **Crisis Detection**: Shows emergency resources when concerning language is detected
- **Memory**: Maintains conversation context using Django cache
- **Privacy**: Everything runs locally - no cloud APIs

## Requirements

- Python 3.11+
- At least 4GB RAM (8GB+ recommended)
- Mac with Apple Silicon or Linux with CUDA (CPU works but slow)

## Setup

```bash
git clone <repository>
cd MentalHealthCounsellingBot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
cd chatbot_project
python manage.py migrate
python manage.py runserver
```

Then go to `http://127.0.0.1:8000/` in your browser.

**Performance:**
- Apple Silicon (M1/M2): ~2-3 seconds per response
- CPU: ~5-10 seconds per response
- Memory usage: ~2-4GB RAM

## API

Simple REST API at `POST /`:

```bash
curl -X POST http://127.0.0.1:8000/ \
  -H 'Content-Type: application/json' \
  -d '{"user_input": "I feel anxious about work."}'
```

Returns:
```json
{
  "reply": "It sounds like work has been overwhelming...",
  "is_crisis": false,
  "session_id": "some-random-id"
}
```

## Troubleshooting

**"Model not found"**: Check that `MODEL_PATH` points to directory with `tokenizer.json` and adapter files.

**Slow responses**: Use MPS/CUDA if available, or reduce `max_new_tokens`.

**Memory errors**: Lower `max_length` or batch size, enable gradient checkpointing.

**CSRF errors**: Set `CSRF_TRUSTED_ORIGINS` for your domain.