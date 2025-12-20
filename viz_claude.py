import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

def create_comprehensive_dashboard(chat_scores, user_scores, chat_history, cached_scores, 
                                   per_chat, composite_score=None, topic="Formula 1",
                                   history_log=None, output_file='dashboard.png'):
    """
    Creates a comprehensive 6-panel dashboard for chat analytics.
    
    Parameters:
    -----------
    chat_scores : dict - Per-chat relevance scores
    user_scores : dict - Per-user composite scores
    chat_history : dict - Full message history
    cached_scores : dict - Cached LLM scores
    per_chat : dict - Per-chat aggregates (weighted_sum, weight_sum, timestamp)
    composite_score : float - Optional global composite
    topic : str - Topic being analyzed
    history_log : list - Historical trend data (optional)
    output_file : str - Output filename
    """
    
    # Create figure with custom grid
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ============ PANEL 1: Top Chat Pairs (Top Left) ============
    ax1 = fig.add_subplot(gs[0, :2])
    
    sorted_chats = sorted(chat_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    if sorted_chats:
        labels = [f"{pair[0]}-{pair[1]}" if isinstance(pair, tuple) else str(pair) 
                  for pair, _ in sorted_chats]
        scores = [score for _, score in sorted_chats]
        
        bars = ax1.barh(range(len(labels)), scores)
        colors = plt.cm.RdYlGn([s/10 for s in scores])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels)
        ax1.set_xlabel('Relevance Score', fontsize=10)
        ax1.set_title(f'Top 10 Chat Pairs - {topic} Relevance', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 10)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
    
    # ============ PANEL 2: User Scores (Top Right) ============
    ax2 = fig.add_subplot(gs[0, 2])
    
    sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)[:8]
    if sorted_users:
        users = [user for user, _ in sorted_users]
        scores = [score for _, score in sorted_users]
        
        bars = ax2.bar(range(len(users)), scores, width=0.6)
        colors = plt.cm.viridis([s/10 for s in scores])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax2.set_xticks(range(len(users)))
        ax2.set_xticklabels(users, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Score', fontsize=10)
        ax2.set_title('Top Users by Engagement', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 10)
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)
    
    # ============ PANEL 3: Network Graph (Middle Left) ============
    ax3 = fig.add_subplot(gs[1, :2])
    
    G = nx.Graph()
    threshold = 3
    for contact, score in chat_scores.items():
        if isinstance(contact, tuple) and score >= threshold:
            user1, user2 = contact
            G.add_edge(user1, user2, weight=score)
    
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*0.7 for w in weights], 
                               alpha=0.5, edge_color=weights, 
                               edge_cmap=plt.cm.RdYlGn, edge_vmin=0, edge_vmax=10, ax=ax3)
        
        # Draw nodes
        node_sizes = [300 + user_scores.get(node, 0) * 50 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                               node_color='lightblue', edgecolors='black', 
                               linewidths=2, ax=ax3)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax3)
        
        ax3.set_title('Chat Network (node size = user engagement)', fontsize=12, fontweight='bold')
        ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for network', ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')
    
    # ============ PANEL 4: Message Timeline (Middle Right) ============
    ax4 = fig.add_subplot(gs[1, 2])
    
    timestamps = []
    scores_list = []
    
    for contact, messages in chat_history.items():
        for msg in messages:
            msg_key = hashlib.sha256(msg['text'].encode()).hexdigest()
            if msg_key in cached_scores.get(contact, {}):
                score = cached_scores[contact][msg_key]
                if score >= 3:
                    timestamps.append(msg['timestamp'])
                    scores_list.append(score)
    
    if timestamps:
        scatter = ax4.scatter(timestamps, scores_list, c=scores_list, 
                             cmap='RdYlGn', s=50, alpha=0.6, 
                             edgecolors='black', linewidth=0.5, vmin=0, vmax=10)
        ax4.set_xlabel('Date', fontsize=9)
        ax4.set_ylabel('Score', fontsize=9)
        ax4.set_title('Message Timeline', fontsize=12, fontweight='bold')
        ax4.set_ylim(0, 10)
        ax4.tick_params(axis='x', rotation=45, labelsize=7)
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Relevance', fraction=0.046, pad=0.04)
    else:
        ax4.text(0.5, 0.5, 'No messages above threshold', ha='center', va='center', transform=ax4.transAxes)
    
    # ============ PANEL 5: Temporal Trends (Bottom Left) ============
    ax5 = fig.add_subplot(gs[2, :2])
    
    if history_log and len(history_log) > 1:
        dates = [entry['date'] for entry in history_log]
        composites = [entry.get('composite', 0) for entry in history_log]
        user_avgs = [entry.get('user_avg', 0) for entry in history_log]
        
        ax5.plot(dates, composites, marker='o', label='Global Composite', 
                linewidth=2, markersize=6)
        ax5.plot(dates, user_avgs, marker='s', label='User Average', 
                linewidth=2, markersize=6, alpha=0.7)
        
        ax5.set_xlabel('Date', fontsize=10)
        ax5.set_ylabel('Score', fontsize=10)
        ax5.set_title('Relevance Trends Over Time', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 10)
        ax5.tick_params(axis='x', rotation=45, labelsize=8)
    else:
        # Generate simulated trend for demo purposes
        days = 14
        dates = [datetime.now() - timedelta(days=days-i) for i in range(days)]
        # Simulate trend based on current composite
        base = composite_score if composite_score else 5.0
        trend = [base + np.random.randn()*0.8 + (i-days/2)*0.1 for i in range(days)]
        trend = [max(0, min(10, t)) for t in trend]
        
        ax5.plot(dates, trend, marker='o', linewidth=2, markersize=5, color='steelblue')
        ax5.set_xlabel('Date', fontsize=10)
        ax5.set_ylabel('Score', fontsize=10)
        ax5.set_title('Relevance Trend (Simulated)', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 10)
        ax5.tick_params(axis='x', rotation=45, labelsize=8)
        ax5.text(0.5, 0.95, 'Add history_log parameter for real trends', 
                transform=ax5.transAxes, ha='center', va='top', 
                fontsize=8, style='italic', color='red')
    
    # ============ PANEL 6: Summary Stats (Bottom Right) ============
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    # Calculate statistics
    total_chats = len([s for s in chat_scores.values() if s > 0])
    total_users = len(user_scores)
    avg_chat_score = np.mean([s for s in chat_scores.values() if s > 0]) if total_chats > 0 else 0
    avg_user_score = np.mean(list(user_scores.values())) if user_scores else 0
    
    total_messages = sum(len(msgs) for msgs in chat_history.values())
    relevant_messages = sum(1 for contact in chat_history 
                           for msg in chat_history[contact]
                           if hashlib.sha256(msg['text'].encode()).hexdigest() 
                           in cached_scores.get(contact, {})
                           and cached_scores[contact][hashlib.sha256(msg['text'].encode()).hexdigest()] >= 3)
    
    stats_text = f"""
    SUMMARY STATISTICS
    ══════════════════════════
    
    Topic: {topic}
    
    Global Composite: {composite_score if composite_score else 'N/A'}
    
    Total Active Chats: {total_chats}
    Total Users: {total_users}
    
    Avg Chat Score: {avg_chat_score:.1f}
    Avg User Score: {avg_user_score:.1f}
    
    Total Messages: {total_messages:,}
    Relevant Messages: {relevant_messages:,}
    Relevance Rate: {100*relevant_messages/total_messages if total_messages > 0 else 0:.1f}%
    
    ══════════════════════════
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Main title
    fig.suptitle(f'{topic} Chat Analytics Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Dashboard saved to {output_file}")
    return fig


# ============ Helper function to log trends over time ============
def log_daily_snapshot(history_log, chat_scores, user_scores, composite_score):
    """
    Append current scores to history_log for temporal tracking.
    Call this daily in your pipeline.
    
    Returns updated history_log
    """
    if history_log is None:
        history_log = []
    
    snapshot = {
        'date': datetime.now(),
        'composite': composite_score if composite_score else 0,
        'user_avg': np.mean(list(user_scores.values())) if user_scores else 0,
        'chat_avg': np.mean([s for s in chat_scores.values() if s > 0]) if chat_scores else 0,
        'active_chats': len([s for s in chat_scores.values() if s > 0]),
        'active_users': len(user_scores)
    }
    
    history_log.append(snapshot)
    return history_log


# ============ Example usage ============
if __name__ == "__main__":
    # Assuming you have these from your pipeline:
    # chat_scores, user_scores, chat_history, cached_scores, per_chat, composite
    
    # Example: Create dashboard
    """
    create_comprehensive_dashboard(
        chat_scores=chat_scores,
        user_scores=user_scores,
        chat_history=chat_history,
        cached_scores=cached_scores,
        per_chat=per_chat,
        composite_score=composite,
        topic="Formula 1",
        history_log=None,  # Pass your history_log if tracking trends
        output_file='f1_dashboard.png'
    )
    """
    
    # Example: Log daily trends
    """
    # In your daily_update_pipeline, add:
    history_log = log_daily_snapshot(
        history_log=history_log,
        chat_scores=updated_chat_scores,
        user_scores=updated_user_scores,
        composite_score=updated_composite
    )
    
    # Save history_log to disk for persistence
    with open('history_log.json', 'w') as f:
        json.dump([{k: v.isoformat() if isinstance(v, datetime) else v 
                    for k, v in entry.items()} 
                   for entry in history_log], f)
    """
    
    print("Dashboard functions ready. Call create_comprehensive_dashboard() with your data.")
