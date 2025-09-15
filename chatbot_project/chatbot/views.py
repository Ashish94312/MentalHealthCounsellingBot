
# chatbot/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import os
import json
import logging
import re

# Hugging Face / Transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Session management for conversation memory
# ----------------------------
from django.core.cache import cache
import uuid

def get_or_create_session(session_id: str = None) -> tuple:
    """Get or create a conversation session"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Get conversation history from cache (expires in 24 hours)
    conversation_key = f"conversation_{session_id}"
    conversation_history = cache.get(conversation_key, [])
    
    return session_id, conversation_history

def save_conversation(session_id: str, user_input: str, bot_response: str):
    """Save conversation to session cache"""
    conversation_key = f"conversation_{session_id}"
    conversation_history = cache.get(conversation_key, [])
    
    # Add new exchange
    conversation_history.append(f"Client: {user_input}")
    conversation_history.append(f"Counselor: {bot_response}")
    
    # Keep only last 10 exchanges (5 Q&A pairs) to manage memory
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    
    # Save with 24-hour expiration
    cache.set(conversation_key, conversation_history, 86400)



# ----------------------------
# Crisis detection setup
# ----------------------------
# SYSTEM_PROMPT = """You are a supportive and professional mental health counselor. You provide empathetic, helpful responses while maintaining appropriate boundaries. You are not a licensed therapist, but you can offer supportive listening and general mental health guidance. Always encourage users to seek professional help when needed."""

SYSTEM_PROMPT = """You are a supportive and professional mental health counselor.
You provide empathetic, helpful responses while maintaining appropriate boundaries.
You are not a licensed therapist, but you can offer supportive listening and general mental health guidance.
Always encourage users to seek professional help when needed.

If the client asks who you are, respond clearly:
"I’m an AI mental health support assistant here to listen and provide supportive guidance."
"""

# Default identity response (can be overridden via cache key "who_are_you_response")
WHO_ARE_YOU_DEFAULT = "I’m an AI mental health support assistant here to listen and provide supportive guidance."

def get_identity_response() -> str:
    """Return the identity response, allowing override from cache."""
    return cache.get("who_are_you_response", WHO_ARE_YOU_DEFAULT)

def is_identity_query(text: str) -> bool:
    """Detect if user is asking about the assistant's identity."""
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
    """Detect if user input contains crisis indicators"""
    text = user_input.lower()
    return any(keyword in text for keyword in CRISIS_KEYWORDS)

def get_crisis_response():
    """Return appropriate crisis response"""
    return {
        'reply': """I'm deeply concerned about what you're sharing with me. Your safety is the most important thing right now.

🚨 **Emergency Resources:**
• **National Suicide Prevention Lifeline: 988** (24/7)
• **Crisis Text Line: Text HOME to 741741**
• **Emergency Services: 911**

Please reach out to a trusted friend, family member, or mental health professional immediately. You are not alone, and support is available right now.""",
        'is_crisis': True
    }

# ----------------------------
# Load TinyLlama fine-tuned model
# ----------------------------
# Get absolute path to the model directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "tinylama-mental-health-mentalchat16k")

# Global variables for lazy loading
tokenizer = None
model = None
device = "cpu"

def load_model():
    """Load model lazily to reduce startup memory usage and select best device."""
    global tokenizer, model, device

    if model is not None:
        return tokenizer, model, device

    # Prefer GPU/MPS if available; otherwise fall back to CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logger.info("Loading fine-tuned TinyLlama model...")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Choose dtype appropriate to device
    if device in ("cuda", "mps"):
        load_dtype = torch.float16
    else:
        load_dtype = torch.float32  # safer on CPU

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=load_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_safetensors=True
    )

    # Move model to device if not already handled
    model.to(device)

    # Optimize model for inference
    model.eval()

    # Optional compile for speed on supported setups
    try:
        if hasattr(torch, "compile") and device in ("cuda", "mps"):
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("✅ Model compiled for faster inference")
    except Exception as compile_error:
        logger.warning(f"Model compile skipped: {compile_error}")

    logger.info(f"✅ Model loaded on {device} with memory optimizations")
    return tokenizer, model, device


def generate_reply(user_input: str, conversation_history: list = None) -> str:
    """Generate a reply from the fine-tuned TinyLlama"""
    
    # Load model lazily on first use
    global_tokenizer, global_model, global_device = load_model()
    
    # Build context-aware prompt
    if conversation_history and len(conversation_history) > 0:
        # Include recent conversation context (last 3 exchanges)
        context = "\n".join(conversation_history[-6:])  # Last 3 Q&A pairs
        prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\nClient: {user_input}\nCounselor:"
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nClient: {user_input}\nCounselor:"
    
    inputs = global_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(global_device)
    
    # Simple generation
    with torch.no_grad():
        outputs = global_model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=global_tokenizer.eos_token_id,
            eos_token_id=global_tokenizer.eos_token_id
        )
        
        response = global_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean and validate the response
    cleaned_response = clean_and_validate_response(response, prompt)
    
    return cleaned_response


def clean_and_validate_response(response: str, original_prompt: str) -> str:
    """Clean and validate the generated response"""
    # Extract only the counselor's response
    if "Counselor:" in response:
        cleaned = response.split("Counselor:")[-1].strip()
    else:
        cleaned = response[len(original_prompt):].strip()
    
    # Remove artifacts
    cleaned = cleaned.replace("Client:", "").strip()
    cleaned = cleaned.replace("User:", "").strip()
    cleaned = cleaned.replace("Human:", "").strip()
    cleaned = cleaned.replace("Assistant:", "").strip()

    # If the model started imagining the next turn, cut at the next speaker label
    speaker_label_pattern = re.compile(r"\n(?:Client|User|Human|Assistant|Counselor):", re.IGNORECASE)
    match = speaker_label_pattern.search(cleaned)
    if match:
        cleaned = cleaned[:match.start()].rstrip()

    # Avoid truncating numbered lists like "1." "2." etc. Keep original text.
    # Light formatting: add newlines before common list markers for readability
    cleaned = cleaned.replace(" 1.", "\n1.")
    cleaned = cleaned.replace(" 2.", "\n2.")
    cleaned = cleaned.replace(" 3.", "\n3.")
    cleaned = cleaned.replace(" 4.", "\n4.")
    cleaned = cleaned.replace(" 5.", "\n5.")
    cleaned = cleaned.replace(" - ", "\n- ")
    
    # Quality checks
    if len(cleaned) < 10:
        cleaned = "I understand you're going through a difficult time. Could you tell me more about what's on your mind?"
    elif len(cleaned) > 1200:
        # Truncate if extremely long, but keep more content to avoid cutting lists
        cleaned = cleaned[:1200].rsplit('\n', 1)[0]
    
    return cleaned

# ----------------------------
# Django View
# ----------------------------
@csrf_exempt
@require_http_methods(["GET", "POST"])
def chatbot_response(request):
    if request.method == 'POST':
        try:
            # Parse input (support JSON + form data)
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

            # Get or create session
            session_id, conversation_history = get_or_create_session(session_id)

            # Crisis check
            if detect_crisis(user_input):
                logger.warning(f"Crisis detected: {user_input[:100]}...")
                return JsonResponse(get_crisis_response())

            # Quick path for identity questions (bypass model)
            if is_identity_query(user_input):
                identity_reply = get_identity_response()
                save_conversation(session_id, user_input, identity_reply)
                return JsonResponse({
                    'reply': identity_reply,
                    'is_crisis': False,
                    'session_id': session_id
                })

            # Generate response with conversation context
            chatbot_reply = generate_reply(user_input, conversation_history)

            # Save conversation to session
            save_conversation(session_id, user_input, chatbot_reply)

            # Log interaction safely
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

    # GET -> serve simple chat UI
    return render(request, 'chatbot/chat.html')
