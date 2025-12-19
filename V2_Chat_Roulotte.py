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

MIN_SCORE_THRESHOLD = 3  # Minimum score to include in aggregations (filters low-relevance noise)

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
        return 5  # Fallback neutral score if parsing fails

def group_messages_by_pair(raw_messages):
    """
    Group raw messages into chat_history using sorted tuple keys for bidirectional consistency.
    """
    chat_history = defaultdict(list)
    for msg in raw_messages:
        key = tuple(sorted([msg["sender"], msg["receiver"]]))  # Sort to normalize direction (e.g., ('a', 'b') always)
        chat_history[key].append({"text": msg["text"], "timestamp": msg["timestamp"]})
    
    for key in chat_history:
        chat_history[key] = sorted(chat_history[key], key=lambda m: m["timestamp"])  # Ensure chronological order
    
    return dict(chat_history)

def process_chat_history(chat_history, topic="Formula 1", half_life_days=7, current_time=None,
                         cached_scores=None, bundle_size=1, bundle_mode='count', compute_global=True):
    """
    Process the full chat history to compute per-chat scores and optional global composite.
    - weighted_sum: Sum of (score * weight) for qualifying items (after threshold).
    - weight_sum: Sum of weights for qualifying items.
    """
    if current_time is None:
        current_time = datetime.now()
    lambda_ = math.log(2) / half_life_days  # Decay rate for exponential weighting (halves every half_life_days)
    
    if cached_scores is None:
        cached_scores = defaultdict(dict)
    
    chat_scores = {}
    global_weighted_sum = 0  # Accumulator for global weighted_sum (if compute_global=True)
    global_weight_sum = 0    # Accumulator for global weight_sum (if compute_global=True)
    per_chat = {}  # Stores per-chat (weighted_sum, weight_sum, last_update_time) for incremental updates
    
    for contact, messages in chat_history.items():
        if not messages:
            continue
        if contact not in cached_scores:
            cached_scores[contact] = {}
        
        chat_weighted_sum = 0  # Per-chat weighted_sum: sum(score * weight) for qualifying bundles/messages
        chat_weight_sum = 0    # Per-chat weight_sum: sum(weight) for qualifying bundles/messages
        
        if bundle_mode == 'daily':
            # Group messages by calendar day for bundling
            daily_groups = defaultdict(list)
            for msg in messages:
                day_key = msg['timestamp'].date()
                daily_groups[day_key].append(msg)
            
            for day, chunk in sorted(daily_groups.items()):
                bundle_text = "\n".join(msg['text'] for msg in chunk)  # Concatenate daily messages
                bundle_key = hashlib.sha256(bundle_text.encode()).hexdigest()  # Hash for caching
                # Average timestamp for the day (to compute age/weight)
                avg_timestamp = sum((msg['timestamp'] - datetime(1970,1,1)).total_seconds() for msg in chunk) / len(chunk)
                avg_timestamp = datetime.fromtimestamp(avg_timestamp)
                age_days = (current_time - avg_timestamp).total_seconds() / 86400  # Fractional days old
                weight = math.exp(-lambda_ * age_days)  # Exponential decay weight (recent = higher)
                
                if bundle_key not in cached_scores[contact]:
                    cached_scores[contact][bundle_key] = score_message_with_llm(bundle_text, topic)
                score = cached_scores[contact][bundle_key]
                
                if score >= MIN_SCORE_THRESHOLD:  # Filter: Only include if relevant enough
                    chat_weighted_sum += score * weight  # Add to per-chat weighted_sum
                    chat_weight_sum += weight            # Add to per-chat weight_sum
        
        elif bundle_mode == 'count' and bundle_size > 1:
            # Fixed-size bundling (chunks of bundle_size messages)
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
            # Per-message mode (no bundling)
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
        
        # Compute per-chat score if qualifying content exists
        if chat_weight_sum > 0:
            chat_scores[contact] = round(chat_weighted_sum / chat_weight_sum, 1)  # Weighted average (1-10 scale)
            if compute_global:
                global_weighted_sum += chat_weighted_sum  # Add to global for composite
                global_weight_sum += chat_weight_sum     # Add to global for composite
            per_chat[contact] = (chat_weighted_sum, chat_weight_sum, current_time)
        else:
            chat_scores[contact] = 0
            per_chat[contact] = (0, 0, current_time)
    
    # Optional global composite: Weighted average across all chats
    composite_score = round(global_weighted_sum / global_weight_sum, 1) if global_weight_sum > 0 and compute_global else None
    return chat_scores, composite_score, cached_scores, global_weighted_sum, global_weight_sum, per_chat

def update_with_new_chat(new_chat_history, chat_history, topic="Formula 1", half_life_days=7, current_time=None,
                         cached_scores=None, old_chat_scores=None, per_chat=None,
                         old_gw_sum=0, old_gw_total=0, bundle_size=1, bundle_mode='count', compute_global=True):
    """
    Update with new chat_history dict (incremental merge and full recalc for affected chats to avoid bug).
    - Merges new into existing.
    - Recomputes entire affected chat (O(n_messages_per_chat)) for correctness in bundling modes.
    - weighted_sum: Sum of (score * weight) for qualifying items.
    - weight_sum: Sum of weights for qualifying items.
    """
    if current_time is None:
        current_time = datetime.now()
    if cached_scores is None or per_chat is None:
        raise ValueError("Provide caches and per_chat")
    
    lambda_ = math.log(2) / half_life_days
    
    # Merge new_chat_history into chat_history
    for contact, new_msgs in new_chat_history.items():
        if contact not in chat_history:
            chat_history[contact] = []
        chat_history[contact].extend(new_msgs)
        chat_history[contact] = sorted(chat_history[contact], key=lambda m: m["timestamp"])  # Re-sort after append
    
    # For each affected contact, recompute the entire chat to avoid double-counting bug
    for contact in new_chat_history:
        old_chat_weighted_sum, old_chat_weight_sum, last_update = per_chat.get(contact, (0, 0, current_time))
        
        # Decay old aggregates (but since full recompute, this is temporary; full calc replaces)
        delta_days = (current_time - last_update).total_seconds() / 86400
        decay = math.exp(-lambda_ * delta_days)
        temp_chat_weighted_sum = old_chat_weighted_sum * decay
        temp_chat_weight_sum = old_chat_weight_sum * decay
        
        # Full recompute for the chat (loop over all bundles/messages)
        messages = chat_history[contact]
        new_chat_weighted_sum = 0
        new_chat_weight_sum = 0
        
        if bundle_mode == 'daily':
            daily_groups = defaultdict(list)
            for msg in messages:
                day_key = msg['timestamp'].date()
                daily_groups[day_key].append(msg)
            
            for day, chunk in sorted(daily_groups.items()):
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
                    new_chat_weighted_sum += score * weight
                    new_chat_weight_sum += weight
        elif bundle_mode == 'count' and bundle_size > 1:
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
                    new_chat_weighted_sum += score * weight
                    new_chat_weight_sum += weight
        else:
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
                    new_chat_weighted_sum += score * weight
                    new_chat_weight_sum += weight
        
        # Update per-chat score
        new_chat_score = round(new_chat_weighted_sum / new_chat_weight_sum, 1) if new_chat_weight_sum > 0 else 0
        old_chat_scores[contact] = new_chat_score
        
        # Update global aggregates (if compute_global=True)
        if compute_global:
            old_gw_sum -= old_chat_weighted_sum
            old_gw_total -= old_chat_weight_sum
            old_gw_sum += new_chat_weighted_sum
            old_gw_total += new_chat_weight_sum
            new_composite = round(old_gw_sum / old_gw_total, 1) if old_gw_total > 0 else 0
        else:
            new_composite = None
        
        per_chat[contact] = (new_chat_weighted_sum, new_chat_weight_sum, current_time)
    
    return old_chat_scores, new_composite, chat_history, cached_scores, old_gw_sum, old_gw_total, per_chat

def compute_user_composites(chat_scores, per_chat, compute_global=True):
    """
    Compute per-user composites as weighted average of their chats' aggregates.
    - Uses per-chat weighted_sum and weight_sum for accuracy.
    - If compute_global, also returns global composite (optional).
    """
    user_composites = defaultdict(lambda: (0, 0))  # Per-user (weighted_sum, weight_sum)
    global_weighted_sum = 0  # For optional global composite
    global_weight_sum = 0    # For optional global composite
    
    for contact, score in chat_scores.items():
        if isinstance(contact, tuple):
            user1, user2 = contact
            chat_weighted_sum, chat_weight_sum, _ = per_chat[contact]
            # Add to user1's aggregates
            u1_weighted_sum, u1_weight_sum = user_composites[user1]
            user_composites[user1] = (u1_weighted_sum + chat_weighted_sum, u1_weight_sum + chat_weight_sum)
            # Add to user2's aggregates
            u2_weighted_sum, u2_weight_sum = user_composites[user2]
            user_composites[user2] = (u2_weighted_sum + chat_weighted_sum, u2_weight_sum + chat_weight_sum)
            
            if compute_global:
                global_weighted_sum += chat_weighted_sum
                global_weight_sum += chat_weight_sum
    
    user_scores = {}
    for user, (weighted_sum, weight_sum) in user_composites.items():
        user_scores[user] = round(weighted_sum / weight_sum, 1) if weight_sum > 0 else 0
    
    global_composite = round(global_weighted_sum / global_weight_sum, 1) if global_weight_sum > 0 and compute_global else None
    return user_scores, global_composite

# Persistence
def save_caches(cached_scores, per_chat):
    serial_per_chat = {str(k): (v[0], v[1], v[2].isoformat()) for k, v in per_chat.items()}
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
            per_chat = {tuple(eval(k)) if k.startswith('(') else k: (v[0], v[1], datetime.fromisoformat(v[2])) for k, v in serial_per_chat.items()}
        return cached_scores, per_chat
    except FileNotFoundError:
        return defaultdict(dict), {}

# Pipeline for initial historical data
def initialize_pipeline(historical_raw_messages, topic="Formula 1", half_life_days=7, bundle_size=1, bundle_mode='count', compute_global=True):
    """
    Initialize with historical data: Group, process, compute scores, save caches.
    Returns: chat_history, chat_scores, composite (or None), user_scores, cached_scores, per_chat, gw_sum, gw_total
    """
    current_time = datetime.now()
    chat_history = group_messages_by_pair(historical_raw_messages)
    cached_scores, per_chat = load_caches()
    chat_scores, _, cached_scores, gw_sum, gw_total, per_chat = process_chat_history(
        chat_history, topic=topic, half_life_days=half_life_days, current_time=current_time,
        cached_scores=cached_scores, bundle_size=bundle_size, bundle_mode=bundle_mode, compute_global=compute_global
    )
    user_scores, composite = compute_user_composites(chat_scores, per_chat, compute_global=compute_global)
    save_caches(cached_scores, per_chat)
    return chat_history, chat_scores, composite, user_scores, cached_scores, per_chat, gw_sum, gw_total

# Daily run pipeline for new chats
def daily_update_pipeline(new_raw_messages, chat_history, old_chat_scores, old_gw_sum, old_gw_total,
                          topic="Formula 1", half_life_days=7, bundle_size=1, bundle_mode='count', compute_global=True):
    """
    Daily update: Group new messages, update incrementally, compute new scores.
    Returns: updated_chat_scores, updated_composite (or None), updated_user_scores, cached_scores, per_chat, new_gw_sum, new_gw_total
    """
    current_time = datetime.now()
    new_chat_history = group_messages_by_pair(new_raw_messages)
    cached_scores, per_chat = load_caches()  # Reload if needed
    
    updated_chat_scores, _, chat_history, cached_scores, new_gw_sum, new_gw_total, per_chat = update_with_new_chat(
        new_chat_history, chat_history, topic=topic, half_life_days=half_life_days, current_time=current_time,
        cached_scores=cached_scores, old_chat_scores=old_chat_scores, per_chat=per_chat,
        old_gw_sum=old_gw_sum, old_gw_total=old_gw_total, bundle_size=bundle_size, bundle_mode=bundle_mode, compute_global=compute_global
    )
    updated_user_scores, updated_composite = compute_user_composites(updated_chat_scores, per_chat, compute_global=compute_global)
    save_caches(cached_scores, per_chat)
    return updated_chat_scores, updated_composite, updated_user_scores, cached_scores, per_chat, new_gw_sum, new_gw_total

# Example usage: Initialize with historical data
historical_raw_messages = [
    {"sender": "Alice", "receiver": "Bob", "text": "Hey Bob, how's it going?", "timestamp": datetime(2025, 12, 1)},
    {"sender": "Bob", "receiver": "Alice", "text": "Good, you?", "timestamp": datetime(2025, 12, 1)},
    {"sender": "Alice", "receiver": "Bob", "text": "Did you watch the F1 race? Verstappen won!", "timestamp": datetime(2025, 12, 2)},
    {"sender": "Bob", "receiver": "Alice", "text": "Yes, amazing!", "timestamp": datetime(2025, 12, 2)},
    {"sender": "Alice", "receiver": "Charlie", "text": "Weather's nice today.", "timestamp": datetime(2025, 12, 3)},
    {"sender": "Charlie", "receiver": "Alice", "text": "Yeah, let's grab coffee.", "timestamp": datetime(2025, 12, 3)},
]

chat_history, chat_scores, composite, user_scores, cached_scores, per_chat, gw_sum, gw_total = initialize_pipeline(
    historical_raw_messages, topic="Formula 1", half_life_days=7, bundle_mode='daily', compute_global=True
)
print("Initial Chat Scores:", chat_scores)
print("Initial Composite (optional):", composite)
print("Initial User Scores:", user_scores)

# Example: Daily update with new messages
new_raw_messages = [
    {"sender": "Alice", "receiver": "Bob", "text": "Another F1 update!", "timestamp": datetime.now()},
    {"sender": "Bob", "receiver": "Alice", "text": "Cool!", "timestamp": datetime.now()},
]

updated_chat_scores, updated_composite, updated_user_scores, cached_scores, per_chat, new_gw_sum, new_gw_total = daily_update_pipeline(
    new_raw_messages, chat_history, old_chat_scores=chat_scores, old_gw_sum=gw_sum, old_gw_total=gw_total,
    topic="Formula 1", half_life_days=7, bundle_mode='daily', compute_global=True
)
print("Updated Chat Scores:", updated_chat_scores)
print("Updated Composite (optional):", updated_composite)
print("Updated User Scores:", updated_user_scores)


# Visualizations
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Sample data from history (simulated for demo)
user_scores = {'Alice': 9.0, 'Bob': 9.0, 'Charlie': 0.0}
chat_scores = {('Alice', 'Bob'): 9.0, ('Alice', 'Charlie'): 0.0}
per_chat = {('Alice', 'Bob'): (3.343, 0.371, datetime.now()), ('Alice', 'Charlie'): (0, 0, datetime.now())}

# Function to visualize top users (bar chart)
def visualize_top_users(user_scores, output_file='top_users.png'):
    # Sort users descending by score
    sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)
    users, scores = zip(*sorted_users) if sorted_users else ([], [])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(users, scores, color='skyblue')
    ax.set_ylim(0, 10)
    ax.set_ylabel('User Composite Score (1-10)')
    ax.set_title('Users with Highest Scores')
    plt.savefig(output_file)
    print(f"Saved top users chart to {output_file}")

# Function for general trends (line chart of average scores over time)
def visualize_general_trends(trend_data, output_file='general_trends.png'):
    # trend_data: dict of {date: average_score}
    # Simulated data
    dates = pd.date_range(start='2025-12-15', periods=5).date
    avg_scores = [4.5, 5.0, 5.5, 6.0, 5.7]  # From example
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dates, avg_scores, marker='o', color='blue')
    ax.set_ylabel('Average User Score')
    ax.set_title('General Trends Across Entire Chats')
    ax.grid(True)
    plt.savefig(output_file)
    print(f"Saved general trends chart to {output_file}")

# Call functions
visualize_top_users(user_scores)
visualize_general_trends(None)
