
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import os
import json
import logging
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from django.core.cache import cache
import uuid

def get_or_create_session(session_id: str = None) -> tuple:
    if not session_id:
        session_id = str(uuid.uuid4())
    
    conversation_key = f"conversation_{session_id}"
    conversation_history = cache.get(conversation_key, [])
    
    return session_id, conversation_history

def save_conversation(session_id: str, user_input: str, bot_response: str):
    conversation_key = f"conversation_{session_id}"
    conversation_history = cache.get(conversation_key, [])
    
    conversation_history.append(f"Client: {user_input}")
    conversation_history.append(f"Counselor: {bot_response}")
    
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    
    cache.set(conversation_key, conversation_history, 86400)

SYSTEM_PROMPT = """You are a supportive and professional mental health counselor.
You provide empathetic, helpful responses while maintaining appropriate boundaries.
You are not a licensed therapist, but you can offer supportive listening and general mental health guidance.
Always encourage users to seek professional help when needed.

If the client asks who you are, respond clearly:
"I’m an AI mental health support assistant here to listen and provide supportive guidance."
"""

WHO_ARE_YOU_DEFAULT = "I’m an AI mental health support assistant here to listen and provide supportive guidance."

def get_identity_response() -> str:
    return cache.get("who_are_you_response", WHO_ARE_YOU_DEFAULT)

def is_identity_query(text: str) -> bool:
    if not text:
        return False
    t = text.lower().strip()
    identity_phrases = [
        "who are you",
        "what are you",
        "who am i talking to",
        "who is this",
        "what is your name",
        "are you human",
        "are you a bot",
        "introduce yourself",
        "tell me about yourself",
    ]
    return any(p in t for p in identity_phrases)

CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'not worth living', 'better off dead',
    'hurt myself', 'self harm', 'cut myself', 'overdose', 'jump off', 'hang myself',
    'shoot myself', 'poison myself', 'drown myself', 'burn myself', 'thinking about ending',
    'want to die', 'end it all', 'not want to live', 'better off without me',
    'feel like hurting', 'want to hurt', 'harm myself', 'self-harm', 'cutting myself'
]

def detect_crisis(user_input: str) -> bool:
    text = user_input.lower()
    return any(keyword in text for keyword in CRISIS_KEYWORDS)

def get_crisis_response():
    return {
        'reply': """I'm deeply concerned about what you're sharing with me. Your safety is the most important thing right now.

🚨 **Emergency Resources:**
• **National Suicide Prevention Lifeline: 988** (24/7)
• **Crisis Text Line: Text HOME to 741741**
• **Emergency Services: 911**

Please reach out to a trusted friend, family member, or mental health professional immediately. You are not alone, and support is available right now.""",
        'is_crisis': True
    }

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "tinylama-mental-health-mentalchat16k")

tokenizer = None
model = None
device = "cpu"

def load_model():
    global tokenizer, model, device

    if model is not None:
        return tokenizer, model, device

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logger.info("Loading fine-tuned TinyLlama model...")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if device in ("cuda", "mps"):
        load_dtype = torch.float16
    else:
        load_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=load_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_safetensors=True
    )

    model.to(device)

    model.eval()

    try:
        if hasattr(torch, "compile") and device in ("cuda", "mps"):
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled for faster inference")
    except Exception as compile_error:
        logger.warning(f"Model compile skipped: {compile_error}")

    logger.info(f"Model loaded on {device} with memory optimizations")
    return tokenizer, model, device

def generate_reply(user_input: str) -> str:

    global_tokenizer, global_model, global_device = load_model()

    prompt = f"{SYSTEM_PROMPT}\n\nClient: {user_input}\nCounselor:"

    inputs = global_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(global_device)

    with torch.no_grad():
        outputs = global_model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.5,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=global_tokenizer.eos_token_id,
            eos_token_id=global_tokenizer.eos_token_id,
        )

        response = global_tokenizer.decode(outputs[0], skip_special_tokens=True)

    cleaned_response = clean_and_validate_response(response, prompt)

    return cleaned_response

def clean_and_validate_response(response: str, original_prompt: str) -> str:
    if "Counselor:" in response:
        cleaned = response.split("Counselor:")[-1].strip()
    else:
        cleaned = response[len(original_prompt):].strip()
    
    cleaned = cleaned.replace("Client:", "").strip()
    cleaned = cleaned.replace("User:", "").strip()
    cleaned = cleaned.replace("Human:", "").strip()
    cleaned = cleaned.replace("Assistant:", "").strip()

    speaker_label_pattern = re.compile(r"\n(?:Client|User|Human|Assistant|Counselor):", re.IGNORECASE)
    match = speaker_label_pattern.search(cleaned)
    if match:
        cleaned = cleaned[:match.start()].rstrip()

    cleaned = cleaned.replace(" 1.", "\n1.")
    cleaned = cleaned.replace(" 2.", "\n2.")
    cleaned = cleaned.replace(" 3.", "\n3.")
    cleaned = cleaned.replace(" 4.", "\n4.")
    cleaned = cleaned.replace(" 5.", "\n5.")
    cleaned = cleaned.replace(" - ", "\n- ")
    
    if len(cleaned) < 10:
        cleaned = "I understand you're going through a difficult time. Could you tell me more about what's on your mind?"
    elif len(cleaned) > 1200:
        cleaned = cleaned[:1200].rsplit('\n', 1)[0]
    
    return cleaned

@csrf_exempt
@require_http_methods(["GET", "POST"])
def chatbot_response(request):
    if request.method == 'POST':
        try:
            if request.content_type == "application/json":
                body = json.loads(request.body.decode('utf-8'))
                user_input = body.get("user_input", "").strip()
                session_id = body.get("session_id", None)
            else:
                user_input = request.POST.get("user_input", "").strip()
                session_id = request.POST.get("session_id", None)

            if not user_input:
                return JsonResponse({
                    'reply': 'Please provide a message so I can help you.',
                    'error': 'Empty input'
                }, status=400)

            session_id, conversation_history = get_or_create_session(session_id)

            if detect_crisis(user_input):
                logger.warning(f"Crisis detected: {user_input[:100]}...")
                return JsonResponse(get_crisis_response())

            if is_identity_query(user_input):
                identity_reply = get_identity_response()
                save_conversation(session_id, user_input, identity_reply)
                return JsonResponse({
                    'reply': identity_reply,
                    'is_crisis': False,
                    'session_id': session_id
                })

            chatbot_reply = generate_reply(user_input)

            save_conversation(session_id, user_input, chatbot_reply)

            logger.info(f"Reply generated for input length {len(user_input)}, session: {session_id[:8]}...")

            return JsonResponse({
                'reply': chatbot_reply,
                'is_crisis': False,
                'session_id': session_id
            })

        except Exception as e:
            logger.error(f"Error in chatbot_response: {str(e)}", exc_info=True)
            return JsonResponse({
                'reply': "I'm experiencing technical issues. Please try again later.",
                'error': 'Internal server error'
            }, status=500)

    return render(request, 'chatbot/chat.html')
