```python
import openai
from collections import defaultdict
import json
import hashlib
from datetime import datetime
import math
import ast  # For secure literal evaluation in loading (safer than eval for parsing tuple keys)

# Set your API key (or load from env)
openai.api_key = "your-api-key-here"
client = openai.OpenAI()  # New syntax for OpenAI library v1+ (replaces deprecated ChatCompletion)

# Prompt template for scoring a single message or bundle
SCORE_PROMPT = """
On a scale of 1 to 10, rate how closely this message bundle relates to the topic of {topic} (1 = completely unrelated, 10 = directly about it). Respond with only the integer score.
Message bundle: {message}
"""

MIN_SCORE_THRESHOLD = 3  # Minimum score to include in aggregations (filters low-relevance noise to strengthen signals)

def score_message_with_llm(message, topic="Formula 1"):
    # Generates a prompt for the LLM to score the message/bundle's relevance to the topic
    prompt = SCORE_PROMPT.format(topic=topic, message=message)
    response = client.chat.completions.create(  # Updated API call for v1+ compatibility
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.0
    )
    try:
        score = int(response.choices[0].message.content.strip())
        return max(1, min(10, score))  # Clamp score to 1-10 range
    except ValueError:
        return 5  # Fallback neutral score if parsing fails (e.g., non-integer response)

def group_messages_by_pair(raw_messages):
    """
    Group raw messages into chat_history using sorted tuple keys for bidirectional consistency.
    This normalizes conversations so A->B and B->A share the same key (e.g., ('A', 'B')).
    """
    chat_history = defaultdict(list)
    for msg in raw_messages:
        key = tuple(sorted([msg["sender"], msg["receiver"]]))  # Sort to normalize direction
        chat_history[key].append({"text": msg["text"], "timestamp": msg["timestamp"]})
    
    for key in chat_history:
        chat_history[key] = sorted(chat_history[key], key=lambda m: m["timestamp"])  # Ensure chronological order for bundling/weighting
    
    return dict(chat_history)

def process_chat_history(chat_history, topic="Formula 1", half_life_days=7, current_time=None,
                         cached_scores=None, bundle_size=1, bundle_mode='count', compute_global=True):
    """
    Process the full chat history to compute per-chat scores and optional global composite.
    - weighted_sum: Sum of (score * weight) for qualifying items (after threshold) - measures total topical "strength" adjusted for recency.
    - weight_sum: Sum of weights for qualifying items - acts as a normalizer for the weighted average, ensuring recency bias is properly scaled.
    These fit into the methodology by allowing efficient incremental updates (stored in per_chat) and user/global aggregations.
    """
    if current_time is None:
        current_time = datetime.now()
    lambda_ = math.log(2) / half_life_days  # Decay rate formula: Measures how quickly relevance fades (e.g., halves influence every half_life_days); fits as the core of recency bias.
    
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
        
        chat_weighted_sum = 0  # Per-chat weighted_sum initialization
        chat_weight_sum = 0    # Per-chat weight_sum initialization
        
        if bundle_mode == 'daily':
            # Group messages by calendar day for bundling (contextual daily relevance)
            daily_groups = defaultdict(list)
            for msg in messages:
                day_key = msg['timestamp'].date()
                daily_groups[day_key].append(msg)
            
            for day, chunk in sorted(daily_groups.items()):
                bundle_text = "\n".join(msg['text'] for msg in chunk)  # Concatenate for LLM input
                bundle_key = hashlib.sha256(bundle_text.encode()).hexdigest()  # Cache key
                # Average timestamp calculation for bundle age
                avg_timestamp = sum((msg['timestamp'] - datetime(1970,1,1)).total_seconds() for msg in chunk) / len(chunk)
                avg_timestamp = datetime.fromtimestamp(avg_timestamp)
                age_days = (current_time - avg_timestamp).total_seconds() / 86400  # Age in fractional days
                weight = math.exp(-lambda_ * age_days)  # Weight formula: Measures recency importance (1 for today, decays exponentially); fits to prioritize "the moment".
                
                if bundle_key not in cached_scores[contact]:
                    cached_scores[contact][bundle_key] = score_message_with_llm(bundle_text, topic)
                score = cached_scores[contact][bundle_key]
                
                if score >= MIN_SCORE_THRESHOLD:  # Filter: Only include if relevant enough
                    chat_weighted_sum += score * weight  # Add to per-chat weighted_sum
                    chat_weight_sum += weight            # Add to per-chat weight_sum
        
        elif bundle_mode == 'count' and bundle_size > 1:
            # Fixed-size bundling for contextual groups
            for i in range(0, len(messages), bundle_size):
                chunk = messages[i:i + bundle_size]
                bundle_text = "\n".join(msg['text'] for msg in chunk)
                bundle_key = hashlib.sha256(bundle_text.encode()).hexdigest()
                avg_timestamp = sum((msg['timestamp'] - datetime(1970,1,1)).total_seconds() for msg in chunk) / len(chunk)
                avg_timestamp = datetime.fromtimestamp(avg_timestamp)
                age_days = (current_time - avg_timestamp).total_seconds() / 86400
                weight = math.exp(-lambda_ * age_days)  # Same weight formula as above
                
                if bundle_key not in cached_scores[contact]:
                    cached_scores[contact][bundle_key] = score_message_with_llm(bundle_text, topic)
                score = cached_scores[contact][bundle_key]
                
                if score >= MIN_SCORE_THRESHOLD:
                    chat_weighted_sum += score * weight
                    chat_weight_sum += weight
        
        else:
            # Per-message mode (granular, no context bundling)
            for msg_dict in messages:
                msg = msg_dict['text']
                timestamp = msg_dict['timestamp']
                age_days = (current_time - timestamp).total_seconds() / 86400
                weight = math.exp(-lambda_ * age_days)  # Weight formula: Ensures recent messages dominate scores; measures temporal relevance decay.
                
                msg_key = hashlib.sha256(msg.encode()).hexdigest()
                if msg_key not in cached_scores[contact]:
                    cached_scores[contact][msg_key] = score_message_with_llm(msg, topic)
                score = cached_scores[contact][msg_key]
                
                if score >= MIN_SCORE_THRESHOLD:
                    chat_weighted_sum += score * weight
                    chat_weight_sum += weight
        
        # Compute per-chat score if qualifying content exists
        if chat_weight_sum > 0:
            chat_scores[contact] = round(chat_weighted_sum / chat_weight_sum, 1)  # Weighted average formula: Measures overall chat relevance (1-10), biased to recent/qualifying content.
            if compute_global:
                global_weighted_sum += chat_weighted_sum  # Add to global for network-wide average
                global_weight_sum += chat_weight_sum     # Add to global normalizer
            per_chat[contact] = (chat_weighted_sum, chat_weight_sum, current_time)
        else:
            chat_scores[contact] = 0
            per_chat[contact] = (0, 0, current_time)
    
    # Optional global composite: Weighted average across all chats
    composite_score = round(global_weighted_sum / global_weight_sum, 1) if global_weight_sum > 0 and compute_global else None  # Global average formula: Measures network-wide topic alignment; fits as optional holistic metric.
    return chat_scores, composite_score, cached_scores, global_weighted_sum, global_weight_sum, per_chat

def update_with_new_chat(new_chat_history, chat_history, topic="Formula 1", half_life_days=7, current_time=None,
                         cached_scores=None, old_chat_scores=None, per_chat=None,
                         bundle_size=1, bundle_mode='count', compute_global=True):
    """
    Update with new chat_history dict (incremental merge and recalc for affected chats).
    - Merges new into existing.
    - Recomputes entire affected chat for correctness in bundling modes (avoids double-counting).
    - weighted_sum: Sum of (score * weight) for qualifying items.
    - weight_sum: Sum of weights for qualifying items.
    """
    if current_time is None:
        current_time = datetime.now()
    if cached_scores is None or per_chat is None:
        raise ValueError("Provide caches and per_chat")
    
    lambda_ = math.log(2) / half_life_days  # Decay rate formula (same as in process)
    
    # Merge new_chat_history into chat_history
    for contact, new_msgs in new_chat_history.items():
        if contact not in chat_history:
            chat_history[contact] = []
        # Assume new_msgs sorted and newer; extend directly for O(k) efficiency
        chat_history[contact].extend(new_msgs)
        # Optional check: Verify assumption (if fails, sort fully)
        if chat_history[contact] and new_msgs and min(m['timestamp'] for m in new_msgs) < max(m['timestamp'] for m in chat_history[contact][:-len(new_msgs)]):
            print(f"Warning: Out-of-order append in {contact} - falling back to full sort")
            chat_history[contact] = sorted(chat_history[contact], key=lambda m: m["timestamp"])
    
    # For each affected contact, recompute the entire chat to ensure correctness
    for contact in new_chat_history:
        messages = chat_history[contact]
        new_chat_weighted_sum = 0  # Reset per-chat weighted_sum
        new_chat_weight_sum = 0    # Reset per-chat weight_sum
        
        # Recompute bundles/messages (same logic as process, but per-chat)
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
                weight = math.exp(-lambda_ * age_days)  # Weight formula (recency measurement)
                
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
        
        per_chat[contact] = (new_chat_weighted_sum, new_chat_weight_sum, current_time)
    
    # Recompute global from scratch (loop all per_chat with decay)
    new_gw_sum = 0
    new_gw_total = 0
    new_composite = None
    if compute_global:
        for contact, (chat_weighted_sum, chat_weight_sum, last_update) in per_chat.items():
            delta_days = (current_time - last_update).total_seconds() / 86400
            decay = math.exp(-lambda_ * delta_days)  # Decay formula for this chat's aggregates (measures time-based fade since last update)
            new_gw_sum += chat_weighted_sum * decay  # Adjusted global weighted_sum contribution
            new_gw_total += chat_weight_sum * decay  # Adjusted global weight_sum normalizer
        new_composite = round(new_gw_sum / new_gw_total, 1) if new_gw_total > 0 else 0  # Global average formula: Measures network-wide topic alignment; fits as optional holistic metric.
    
    return old_chat_scores, new_composite, chat_history, cached_scores, new_gw_sum, new_gw_total, per_chat

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
        user_scores[user] = round(weighted_sum / weight_sum, 1) if weight_sum > 0 else 0  # User average formula: Measures individual's topic engagement across their chats; fits as personalized trend/bias metric.
    
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
            per_chat = {}
            for k, v in serial_per_chat.items():
                key = tuple(ast.literal_eval(k)) if k.startswith('(') else k  # Secure parsing of tuple keys (avoids eval risks)
                per_chat[key] = (v[0], v[1], datetime.fromisoformat(v[2]))
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
    chat_scores, composite, cached_scores, gw_sum, gw_total, per_chat = process_chat_history(
        chat_history, topic=topic, half_life_days=half_life_days, current_time=current_time,
        cached_scores=cached_scores, bundle_size=bundle_size, bundle_mode=bundle_mode, compute_global=compute_global
    )
    user_scores, _ = compute_user_composites(chat_scores, per_chat, compute_global=False)  # Focus on users; global optional
    save_caches(cached_scores, per_chat)
    return chat_history, chat_scores, composite, user_scores, cached_scores, per_chat, gw_sum, gw_total

# Daily run pipeline for new chats
def daily_update_pipeline(new_raw_messages, chat_history, old_chat_scores,
                          topic="Formula 1", half_life_days=7, bundle_size=1, bundle_mode='count', compute_global=True):
    """
    Daily update: Group new messages, update incrementally, compute new scores.
    Returns: updated_chat_scores, updated_composite (or None), updated_user_scores, cached_scores, per_chat, new_gw_sum, new_gw_total
    """
    current_time = datetime.now()
    new_chat_history = group_messages_by_pair(new_raw_messages)
    cached_scores, per_chat = load_caches()  # Reload if needed
    
    updated_chat_scores, updated_composite, chat_history, cached_scores, new_gw_sum, new_gw_total, per_chat = update_with_new_chat(
        new_chat_history, chat_history, topic=topic, half_life_days=half_life_days, current_time=current_time,
        cached_scores=cached_scores, old_chat_scores=old_chat_scores, per_chat=per_chat,
        bundle_size=bundle_size, bundle_mode=bundle_mode, compute_global=compute_global
    )
    updated_user_scores, _ = compute_user_composites(updated_chat_scores, per_chat, compute_global=False)  # Focus on users
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
    new_raw_messages, chat_history, old_chat_scores=chat_scores,
    topic="Formula 1", half_life_days=7, bundle_mode='daily', compute_global=True
)
print("Updated Chat Scores:", updated_chat_scores)
print("Updated Composite (optional):", updated_composite)
print("Updated User Scores:", updated_user_scores)
```
# Viz

import matplotlib.pyplot as plt
import pandas as pd

def visualize_top_chats(chat_scores, top_n=10, output_file='top_chats.png'):
    """Bar chart of highest-scoring chat pairs"""
    # Sort by score descending
    sorted_chats = sorted(chat_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Format labels (tuple to "Alice-Bob")
    labels = [f"{pair[0]}-{pair[1]}" if isinstance(pair, tuple) else str(pair) 
              for pair, _ in sorted_chats]
    scores = [score for _, score in sorted_chats]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(labels, scores, color='skyblue')
    
    # Color-code by score intensity
    colors = plt.cm.RdYlGn([s/10 for s in scores])
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Relevance Score (1-10)')
    ax.set_title(f'Top {top_n} Chat Pairs by Topic Relevance')
    ax.set_xlim(0, 10)
    ax.invert_yaxis()  # Highest on top
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved to {output_file}")
def visualize_user_scores(user_scores, top_n=15, output_file='user_scores.png'):
    """Bar chart of user composite scores"""
    sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    users = [user for user, _ in sorted_users]
    scores = [score for _, score in sorted_users]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(users, scores)
    
    # Gradient coloring
    colors = plt.cm.viridis([s/10 for s in scores])
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_ylabel('User Composite Score (1-10)')
    ax.set_title('Users Ranked by Topic Engagement')
    ax.set_ylim(0, 10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved to {output_file}")

def visualize_temporal_trends(history_log, output_file='trends_over_time.png'):
    """
    Line chart showing composite/user scores over time.
    history_log: list of dicts like [{'date': datetime, 'composite': 7.5, 'user_avg': 6.2}, ...]
    """
    df = pd.DataFrame(history_log)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['date'], df['composite'], marker='o', label='Global Composite', linewidth=2)
    ax.plot(df['date'], df['user_avg'], marker='s', label='User Average', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Score (1-10)')
    ax.set_title('Topic Relevance Trends Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved to {output_file}")

import networkx as nx

def visualize_chat_network(chat_scores, threshold=3, output_file='chat_network.png'):
    """Network graph where nodes=users, edges=chats, thickness=score"""
    G = nx.Graph()
    
    for contact, score in chat_scores.items():
        if isinstance(contact, tuple) and score >= threshold:
            user1, user2 = contact
            G.add_edge(user1, user2, weight=score)
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw edges with width proportional to score
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*0.5 for w in weights], 
                           alpha=0.6, edge_color=weights, 
                           edge_cmap=plt.cm.RdYlGn, edge_vmin=0, edge_vmax=10)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', 
                           edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    ax.set_title('Chat Network by Topic Relevance')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved to {output_file}")
import numpy as np
import seaborn as sns

def visualize_chat_heatmap(chat_scores, output_file='chat_heatmap.png'):
    """Heatmap showing chat scores between all user pairs"""
    # Extract unique users
    users = set()
    for contact in chat_scores.keys():
        if isinstance(contact, tuple):
            users.update(contact)
    users = sorted(list(users))
    
    # Build matrix
    n = len(users)
    matrix = np.zeros((n, n))
    user_idx = {user: i for i, user in enumerate(users)}
    
    for contact, score in chat_scores.items():
        if isinstance(contact, tuple):
            i, j = user_idx[contact[0]], user_idx[contact[1]]
            matrix[i][j] = score
            matrix[j][i] = score  # Symmetric
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, xticklabels=users, yticklabels=users, 
                annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=10,
                cbar_kws={'label': 'Relevance Score'}, ax=ax)
    ax.set_title('Chat Relevance Heatmap')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved to {output_file}")
def visualize_message_timeline(chat_history, cached_scores, topic="Formula 1", 
                                threshold=3, output_file='message_timeline.png'):
    """Scatter plot of relevant messages over time"""
    timestamps = []
    scores_list = []
    
    for contact, messages in chat_history.items():
        for msg in messages:
            msg_key = hashlib.sha256(msg['text'].encode()).hexdigest()
            if msg_key in cached_scores.get(contact, {}):
                score = cached_scores[contact][msg_key]
                if score >= threshold:
                    timestamps.append(msg['timestamp'])
                    scores_list.append(score)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    scatter = ax.scatter(timestamps, scores_list, c=scores_list, 
                         cmap='RdYlGn', s=100, alpha=0.6, 
                         edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Message Relevance Score')
    ax.set_title(f'Timeline of {topic}-Related Messages')
    ax.set_ylim(0, 10)
    plt.colorbar(scatter, label='Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved to {output_file}")
import plotly.graph_objects as go

def visualize_conversation_flow(chat_history, cached_scores, threshold=5):
    """Sankey diagram of message flows (requires plotly)"""
    sources, targets, values, colors = [], [], [], []
    user_set = set()
    
    for contact, messages in chat_history.items():
        if not isinstance(contact, tuple):
            continue
        user1, user2 = contact
        user_set.update([user1, user2])
        
        # Count relevant messages
        count = sum(1 for msg in messages 
                   if hashlib.sha256(msg['text'].encode()).hexdigest() 
                   in cached_scores.get(contact, {})
                   and cached_scores[contact][hashlib.sha256(msg['text'].encode()).hexdigest()] >= threshold)
        
        if count > 0:
            user_idx = {user: i for i, user in enumerate(sorted(user_set))}
            sources.append(user_idx[user1])
            targets.append(user_idx[user2])
            values.append(count)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=sorted(user_set)),
        link=dict(source=sources, target=targets, value=values)
    )])
    
    fig.update_layout(title="Message Flow (Relevant Conversations)")
    fig.write_html("conversation_flow.html")
    print("Saved to conversation_flow.html")

# After running your pipeline:
visualize_top_chats(chat_scores, top_n=15)
visualize_user_scores(user_scores, top_n=20)
visualize_chat_network(chat_scores, threshold=4)
visualize_chat_heatmap(chat_scores)
visualize_message_timeline(chat_history, cached_scores)

# For trends, you'd need to log data daily:
# history_log.append({'date': datetime.now(), 
#                     'composite': composite, 
#                     'user_avg': np.mean(list(user_scores.values()))})
# visualize_temporal_trends(history_log)
