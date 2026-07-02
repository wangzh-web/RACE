"""
Visualization utilities for paper figures
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os


def plot_tradeoff(results: Dict[str, tuple], save_path: str = "results/figures/tradeoff.pdf"):
    """
    Cost vs Accuracy Trade-off     
    results: {"Method Name": (cost, accuracy), ...}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    markers = {
        'All-Edge': 's', 
        'All-Cloud': 's', 
        'Random-50': 'o',
        'Random-70': 'o',
        'Prior Art': '^', 
        'TECS (Ours)': '*',
        'Ours': '*'
    }
    
    colors = {
        'All-Edge': 'gray', 
        'All-Cloud': 'gray',
        'Random-50': 'blue', 
        'Random-70': 'lightblue',
        'Prior Art': 'orange', 
        'TECS (Ours)': 'red',
        'Ours': 'red'
    }
    
    for method, (cost, acc) in results.items():
        size = 300 if 'Ours' in method or 'TECS' in method else 150
        ax.scatter(cost, acc, 
                  marker=markers.get(method, 'o'),
                  c=colors.get(method, 'black'), 
                  s=size, 
                  label=method,
                  edgecolors='black',
                  linewidths=1)
    
    for method, (cost, acc) in results.items():
        if 'Ours' in method or 'TECS' in method:
            ax.annotate('Sweet Spot', 
                       xy=(cost, acc), 
                       xytext=(cost + 0.005, acc - 0.08),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=12,
                       color='red')
    
    ax.set_xlabel('Average Cost per Request ($)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Cost-Accuracy Trade-off', fontsize=14)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {save_path}")
    
    return fig, ax


def plot_threshold_adaptation(threshold_history: List[tuple], 
                             save_path: str = "results/figures/threshold_adaptation.pdf"):
    """
    Online Calibration     
    threshold_history: [(request_count, threshold), ...]
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if threshold_history:
        requests, thresholds = zip(*threshold_history)
        ax.plot(requests, thresholds, 'b-', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Number of Requests', fontsize=12)
    ax.set_ylabel('Conformal Threshold τ', fontsize=12)
    ax.set_title('Online Conformal Calibration: Threshold Adaptation', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {save_path}")
    
    return fig, ax


def plot_risk_budget(budget_history: List[Dict], 
                    save_path: str = "results/figures/risk_budget.pdf"):
    """
        """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if budget_history:
        counts = [h['period_count'] for h in budget_history]
        risks = [h['current_risk'] for h in budget_history]
        remaining = [h['remaining_budget'] for h in budget_history]
        
        ax.fill_between(counts, 0, risks, alpha=0.3, color='red', label='Used Budget')
        ax.fill_between(counts, risks, [r+re for r, re in zip(risks, remaining)], 
                       alpha=0.3, color='green', label='Remaining Budget')
        ax.plot(counts, risks, 'r-', linewidth=2)
    
    ax.set_xlabel('Request Count in Period', fontsize=12)
    ax.set_ylabel('Risk Budget', fontsize=12)
    ax.set_title('Semantic Risk Budget Allocation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {save_path}")
    
    return fig, ax


def create_results_table(results: Dict[str, Dict]) -> str:
    """
    LaTeX     """
    header = r"""
\begin{table}[t]
\centering
\caption{Experimental Results on MMLU Dataset}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
Method & Accuracy$\uparrow$ & Avg Cost$\downarrow$ & Edge Ratio & Eff. Throughput \\
\midrule
"""
    
    rows = []
    for method, metrics in results.items():
        acc = metrics.get('accuracy', 0) * 100
        cost = metrics.get('avg_cost', 0)
        edge = metrics.get('edge_ratio', 0) * 100
        eff = acc  # Simplified: effective throughput ≈ accuracy for fixed N
        
        # Bold for our method
        if 'Ours' in method or 'TECS' in method:
            row = f"\\textbf{{{method}}} & \\textbf{{{acc:.1f}\\%}} & \\textbf{{\\${cost:.4f}}} & {edge:.0f}\\% & {eff:.1f} \\\\"
        else:
            row = f"{method} & {acc:.1f}\\% & \\${cost:.4f} & {edge:.0f}\\% & {eff:.1f} \\\\"
        rows.append(row)
    
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return header + "\n".join(rows) + footer


if __name__ == "__main__":
    example_results = {
        "All-Edge": (0.000, 0.52),
        "All-Cloud": (0.030, 0.95),
        "Random-50": (0.015, 0.74),
        "TECS (Ours)": (0.008, 0.90),
    }
    
    plot_tradeoff(example_results)
    print("Example tradeoff plot created!")
