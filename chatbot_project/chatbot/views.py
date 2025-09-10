
# chatbot/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import os
import json
import logging

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

logger.info("Loading fine-tuned TinyLlama model...")
# Use only MPS device as preferred
device = "mps" if torch.backends.mps.is_available() else None
if device is None:
    raise RuntimeError("MPS device is not available. This application requires MPS support.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

# Optimize model for inference speed
model.eval()
if hasattr(torch, 'compile') and device == "mps":
    try:
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("✅ Model compiled for faster inference")
    except Exception as e:
        logger.warning(f"Model compilation failed: {e}")

logger.info(f"✅ Model loaded on {device}")


def generate_reply(user_input: str, conversation_history: list = None) -> str:
    """Generate a reply from the fine-tuned TinyLlama"""
    
    # Build context-aware prompt
    if conversation_history and len(conversation_history) > 0:
        # Include recent conversation context (last 3 exchanges)
        context = "\n".join(conversation_history[-6:])  # Last 3 Q&A pairs
        prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\nClient: {user_input}\nCounselor:"
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nClient: {user_input}\nCounselor:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Simple generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
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
    cleaned = cleaned.replace("Human:", "").strip()
    cleaned = cleaned.replace("Assistant:", "").strip()
    
    # Remove incomplete sentences at the end
    sentences = cleaned.split('.')
    if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
        cleaned = '.'.join(sentences[:-1]) + '.'
    
    # Ensure proper ending
    if not cleaned.endswith(('.', '!', '?')):
        cleaned += '.'
    
    # Quality checks
    if len(cleaned) < 10:
        cleaned = "I understand you're going through a difficult time. Could you tell me more about what's on your mind?"
    elif len(cleaned) > 400:
        # Truncate if too long
        sentences = cleaned.split('.')
        cleaned = '.'.join(sentences[:3]) + '.'
    
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
