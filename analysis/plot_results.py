"""
Generate performance comparison charts for the RAG system.

Run: python analysis/plot_results.py
Outputs: analysis/figures/
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_accuracy_comparison():
    """Plot accuracy by approach and query type."""

    approaches = ['Semantic', 'Hybrid', 'HyDE']

    # Results from evaluation
    easy = [90, 90, 90]
    vague = [50, 60, 70]
    hard = [90, 100, 90]
    overall = [77, 83, 83]

    x = np.arange(len(approaches))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - 1.5*width, easy, width, label='Easy', color='#4CAF50')
    bars2 = ax.bar(x - 0.5*width, vague, width, label='Vague', color='#FFC107')
    bars3 = ax.bar(x + 0.5*width, hard, width, label='Hard', color='#2196F3')
    bars4 = ax.bar(x + 1.5*width, overall, width,
                   label='Overall', color='#9C27B0')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('RAG Retrieval Accuracy by Approach and Query Type')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()
    ax.set_ylim(0, 110)

    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save
    output_dir = Path('analysis/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    print(f'Saved: {output_dir}/accuracy_comparison.png')


def plot_latency_tradeoff():
    """Plot latency vs accuracy tradeoff."""

    # Data points (accuracy%, latency_ms)
    data = {
        'Semantic': (77, 310),
        'Hybrid': (83, 350),
        'HyDE': (83, 1800),
        'Reranking': (83, 6000),
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (acc, lat) in data.items():
        color = '#4CAF50' if lat < 1000 else '#FFC107' if lat < 3000 else '#F44336'
        ax.scatter(lat, acc, s=200, label=name, c=color, edgecolors='black')
        ax.annotate(name, (lat, acc), textcoords="offset points",
                    xytext=(10, 5), fontsize=10)

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Latency Tradeoff')
    ax.set_xlim(0, 7000)
    ax.set_ylim(70, 90)
    ax.axvline(x=1000, color='gray', linestyle='--',
               alpha=0.5, label='1s threshold')

    plt.tight_layout()

    output_dir = Path('analysis/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'latency_tradeoff.png', dpi=150)
    print(f'Saved: {output_dir}/latency_tradeoff.png')


def plot_vague_query_improvement():
    """Show how vague query accuracy improved."""

    approaches = ['Baseline\n(Semantic)',
                  '+Hybrid\nSearch', '+HyDE\nExpansion']
    accuracy = [50, 60, 70]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#F44336', '#FFC107', '#4CAF50']
    bars = ax.bar(approaches, accuracy, color=colors, edgecolor='black')

    ax.set_ylabel('Accuracy on Vague Queries (%)')
    ax.set_title('Improving Vague Query Performance')
    ax.set_ylim(0, 100)

    # Add improvement annotations
    ax.annotate('', xy=(1, 60), xytext=(0, 50),
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('+10%', xy=(0.5, 55), fontsize=10, ha='center')

    ax.annotate('', xy=(2, 70), xytext=(1, 60),
                arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('+10%', xy=(1.5, 65), fontsize=10, ha='center')

    # Add value labels
    for bar, acc in zip(bars, accuracy):
        ax.annotate(f'{acc}%', xy=(bar.get_x() + bar.get_width()/2, acc),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_dir = Path('analysis/figures')
    plt.savefig(output_dir / 'vague_improvement.png', dpi=150)
    print(f'Saved: {output_dir}/vague_improvement.png')


if __name__ == '__main__':
    print('Generating performance charts...\n')

    plot_accuracy_comparison()
    plot_latency_tradeoff()
    plot_vague_query_improvement()

    print('\nDone! Check analysis/figures/')
