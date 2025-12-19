```python
import openai
from collections import defaultdict
import json
import hashlib
from datetime import datetime
import math

# Set your API key (or load from env)
openai.api_key = "your-api-key-here"

# Prompt template for scoring a single message or bundle
SCORE_PROMPT = """
On a scale of 1 to 10, rate how closely this message bundle relates to the topic of {topic} (1 = completely unrelated, 10 = directly about it). Respond with only the integer score.
Message bundle: {message}
"""

MIN_SCORE_THRESHOLD = 3  # Minimum score to include in aggregations

def score_message_with_llm(message, topic="Formula 1"):
    prompt = SCORE_PROMPT.format(topic=topic, message=message)
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.0
    )
    try:
        score = int(response.choices[0].message.content.strip())
        return max(1, min(10, score))
    except ValueError:
        return 5

def process_chat_history(chat_history, topic="Formula 1", half_life_days=7, current_time=None,
                         cached_scores=None, bundle_size=1):
    """
    Updated with bundle_size (default 1: no bundling).
    per_chat: {contact: (chat_weighted_sum, chat_weight_sum, last_update_time)}
    """
    if current_time is None:
        current_time = datetime.now()
    lambda_ = math.log(2) / half_life_days
    
    if cached_scores is None:
        cached_scores = defaultdict(dict)
    
    chat_scores = {}
    global_weighted_sum = 0
    global_weight_sum = 0
    per_chat = {}
    
    for contact, messages in chat_history.items():
        if not messages:
            continue
        if contact not in cached_scores:
            cached_scores[contact] = {}
        
        chat_weighted_sum = 0
        chat_weight_sum = 0
        
        # Bundle if bundle_size > 1
        if bundle_size > 1:
            for i in range(0, len(messages), bundle_size):
                chunk = messages[i:i + bundle_size]
                bundle_text = "\n".join(msg['text'] for msg in chunk)
                bundle_key = hashlib.sha256(bundle_text.encode()).hexdigest()
                avg_timestamp = sum((msg['timestamp'] - datetime(1970,1,1)).total_seconds() for msg in chunk) / len(chunk)
                avg_timestamp = datetime.fromtimestamp(avg_timestamp)
                age_days = (current_time - avg_timestamp).total_seconds() / 86400
                weight = math.exp(-lambda_ * age_days)
                
                if bundle_key not in cached_scores[contact]:
                    cached_scores[contact][bundle_key] = score_message_with_llm(bundle_text, topic)
                score = cached_scores[contact][bundle_key]
                
                if score >= MIN_SCORE_THRESHOLD:
                    chat_weighted_sum += score * weight
                    chat_weight_sum += weight
        else:
            # Original per-message loop
            for msg_dict in messages:
                msg = msg_dict['text']
                timestamp = msg_dict['timestamp']
                age_days = (current_time - timestamp).total_seconds() / 86400
                weight = math.exp(-lambda_ * age_days)
                
                msg_key = hashlib.sha256(msg.encode()).hexdigest()
                if msg_key not in cached_scores[contact]:
                    cached_scores[contact][msg_key] = score_message_with_llm(msg, topic)
                score = cached_scores[contact][msg_key]
                
                if score >= MIN_SCORE_THRESHOLD:
                    chat_weighted_sum += score * weight
                    chat_weight_sum += weight
        
        if chat_weight_sum > 0:
            chat_scores[contact] = round(chat_weighted_sum / chat_weight_sum, 1)
            global_weighted_sum += chat_weighted_sum
            global_weight_sum += chat_weight_sum
            per_chat[contact] = (chat_weighted_sum, chat_weight_sum, current_time)
        else:
            chat_scores[contact] = 0
            per_chat[contact] = (0, 0, current_time)
    
    composite_score = round(global_weighted_sum / global_weight_sum, 1) if global_weight_sum > 0 else 0
    return chat_scores, composite_score, cached_scores, global_weighted_sum, global_weight_sum, per_chat

def update_with_new_message(contact, new_message, chat_history, topic="Formula 1", half_life_days=7, current_time=None,
                            cached_scores=None, old_chat_scores=None, per_chat=None,
                            old_gw_sum=0, old_gw_total=0, bundle_size=1):
    if current_time is None:
        current_time = datetime.now()
    if cached_scores is None or per_chat is None:
        raise ValueError("Provide caches and per_chat")
    
    lambda_ = math.log(2) / half_life_days
    
    # Append new message
    new_timestamp = current_time
    if contact not in chat_history:
        chat_history[contact] = []
    chat_history[contact].append({"text": new_message, "timestamp": new_timestamp})
    
    # Get old aggregates
    old_chat_sum, old_chat_total, last_update = per_chat.get(contact, (0, 0, current_time))
    
    # Decay old sums
    delta_days = (current_time - last_update).total_seconds() / 86400
    decay = math.exp(-lambda_ * delta_days)
    new_chat_weighted_sum = old_chat_sum * decay
    new_chat_weight_sum = old_chat_total * decay
    
    # Handle new addition with bundling
    if bundle_size > 1:
        # Re-bundle only the last incomplete bundle + new
        messages = chat_history[contact]
        last_bundle_start = max(0, len(messages) - bundle_size)
        chunk = messages[last_bundle_start:]
        bundle_text = "\n".join(msg['text'] for msg in chunk)
        bundle_key = hashlib.sha256(bundle_text.encode()).hexdigest()
        avg_timestamp = sum((msg['timestamp'] - datetime(1970,1,1)).total_seconds() for msg in chunk) / len(chunk)
        avg_timestamp = datetime.fromtimestamp(avg_timestamp)
        age_days = (current_time - avg_timestamp).total_seconds() / 86400
        new_weight = math.exp(-lambda_ * age_days)
        
        if bundle_key not in cached_scores[contact]:
            cached_scores[contact][bundle_key] = score_message_with_llm(bundle_text, topic)
        new_score = cached_scores[contact][bundle_key]
        
        if new_score >= MIN_SCORE_THRESHOLD:
            new_chat_weighted_sum += new_score * new_weight
            new_chat_weight_sum += new_weight
    else:
        # Original: Add new individually
        msg_key = hashlib.sha256(new_message.encode()).hexdigest()
        if msg_key not in cached_scores[contact]:
            cached_scores[contact][msg_key] = score_message_with_llm(new_message, topic)
        new_score = cached_scores[contact][msg_key]
        if new_score >= MIN_SCORE_THRESHOLD:
            new_chat_weighted_sum += new_score * 1  # weight=1 for new
            new_chat_weight_sum += 1
    
    # Update score and globals
    new_chat_score = round(new_chat_weighted_sum / new_chat_weight_sum, 1) if new_chat_weight_sum > 0 else 0
    old_chat_scores[contact] = new_chat_score
    
    new_gw_sum = old_gw_sum - old_chat_sum + new_chat_weighted_sum
    new_gw_total = old_gw_total - old_chat_total + new_chat_weight_sum
    new_composite = round(new_gw_sum / new_gw_total, 1) if new_gw_total > 0 else 0
    
    per_chat[contact] = (new_chat_weighted_sum, new_chat_weight_sum, current_time)
    
    return old_chat_scores, new_composite, chat_history, cached_scores, new_gw_sum, new_gw_total, per_chat

# Persistence (unchanged, but bundles use concat hashes)
def save_caches(cached_scores, per_chat):
    serial_per_chat = {k: (v[0], v[1], v[2].isoformat()) for k, v in per_chat.items()}
    with open("cached_scores_llm.json", "w") as f:
        json.dump(cached_scores, f)
    with open("per_chat.json", "w") as f:
        json.dump(serial_per_chat, f)

def load_caches():
    try:
        with open("cached_scores_llm.json", "r") as f:
            cached_scores = defaultdict(dict, json.load(f))
        with open("per_chat.json", "r") as f:
            serial_per_chat = json.load(f)
            per_chat = {k: (v[0], v[1], datetime.fromisoformat(v[2])) for k, v in serial_per_chat.items()}
        return cached_scores, per_chat
    except FileNotFoundError:
        return defaultdict(dict), {}

# Example usage with bundling
current_time = datetime(2025, 12, 17)
chat_history = {
    "FriendA": [
        {"text": "Hey, how's it going?", "timestamp": datetime(2025, 12, 1)},
        {"text": "Did you watch the F1 race yesterday? Verstappen won!", "timestamp": datetime(2025, 12, 16)}
    ],
    "FriendB": [
        {"text": "Let's grab coffee.", "timestamp": datetime(2025, 12, 10)},
        {"text": "Weather's nice today.", "timestamp": datetime(2025, 12, 15)}
    ]
}
cached_scores, per_chat = load_caches()
chat_scores, composite, cached_scores, gw_sum, gw_total, per_chat = process_chat_history(
    chat_history, half_life_days=7, current_time=current_time, cached_scores=cached_scores, bundle_size=1  # or 5 for bundling
)
print("LLM Version - Chat scores:", chat_scores)
print("LLM Version - Composite score:", composite)
save_caches(cached_scores, per_chat)

# Update example
new_msg = "Hamilton is retiring from F1 next year."
updated_chat_scores, updated_composite, _, cached_scores, new_gw_sum, new_gw_total, per_chat = update_with_new_message(
    "FriendA", new_msg, chat_history, half_life_days=7, current_time=current_time,
    cached_scores=cached_scores, old_chat_scores=chat_scores, per_chat=per_chat,
    old_gw_sum=gw_sum, old_gw_total=gw_total, bundle_size=1  # or 5
)
print("LLM Version - Updated chat scores:", updated_chat_scores)
print("LLM Version - Updated composite:", updated_composite)
save_caches(cached_scores, per_chat)
```
