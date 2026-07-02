"""Regenerate the 3 data-driven figures (Pareto, edge-error, cross-model) with
canonical + GPT-5(low) numbers. Reuses the plotting style of regenerate_all_figures.

cross-model is restricted to the two rerun edge models (Phi-3, Mistral) to stay
consistent with the multi-model table; LLaMA is dropped (not rerun under the
unified protocol). Annotations show CIR(%).
"""
import regenerate_all_figures as R

plt, np, NC = R.plt, R.np, R.NC


def S(acc, cir, err):
    return {'accuracy': acc, 'accuracy_std': 0.0, 'cost': cir, 'cost_std': 0.0,
            'edge_error': err, 'edge_error_std': 0.0}


stats = {
    'phi3': {
        'mmlu': {
            'all_cloud': S(93.40, 100.00, 0.00), 'all_edge': S(69.15, 0.00, 30.85),
            'random_0.5': S(81.12, 49.63, 30.87), 'latency_first': S(69.15, 0.00, 30.85),
            'static': S(75.35, 15.00, 26.94), 'aci': S(77.65, 18.45, 24.95),
            'frugalgpt': S(77.80, 19.40, 24.88), 'larc': S(76.45, 16.20, 25.89),
            'reco': S(71.95, 6.25, 29.33),
        },
        'triviaqa': {
            'all_cloud': S(93.30, 100.00, 0.00), 'all_edge': S(70.10, 0.00, 29.90),
            'random_0.5': S(81.63, 49.63, 30.55), 'latency_first': S(70.10, 0.00, 29.90),
            'static': S(76.25, 10.25, 25.85), 'aci': S(76.60, 11.10, 25.48),
            'frugalgpt': S(78.80, 15.55, 23.92), 'larc': S(74.70, 7.35, 26.82),
            'reco': S(72.85, 4.05, 28.09),
        },
    },
    'mistral': {
        'mmlu': {
            'all_edge': S(53.50, 0.00, 46.50), 'static': S(78.40, 45.45, 33.91),
            'aci': S(84.30, 58.55, 27.02), 'reco': S(82.10, 51.85, 30.74),
        },
    },
}

R.fig_cost_accuracy(stats, 'mmlu')
R.fig_cost_accuracy(stats, 'triviaqa')
R.fig_edge_error(stats)


def cross_model_2(stats, dataset='mmlu'):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    models = ['phi3', 'mistral']
    model_labels = ['Phi-3-medium', 'Mistral-7B']
    methods = ['all_edge', 'static', 'aci', 'reco']
    method_labels = ['Edge-Only', 'Static', 'ACI', 'RACE']
    bar_colors = [NC['red'], NC['cyan'], NC['green'], NC['blue']]
    x = np.arange(len(models))
    width = 0.20
    for i, (m, ml, c) in enumerate(zip(methods, method_labels, bar_colors)):
        accs, cirs = [], []
        for mdl in models:
            d = stats.get(mdl, {}).get(dataset, {}).get(m)
            accs.append(d['accuracy'] if d else 0)
            cirs.append(d['cost'] if d else None)
        bars = ax.bar(x + i * width, accs, width, label=ml, color=c)
        for xi, a, ci in zip(x + i * width, accs, cirs):
            if a > 0 and ci is not None:
                ax.annotate(f'{ci:.0f}', (xi, a), textcoords='offset points',
                            xytext=(0, 1.5), ha='center', va='bottom', fontsize=5.5)
    ax.set_xlabel('Edge Model')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(35, 98)
    ax.legend(loc='upper center', ncol=4, frameon=True, framealpha=0.9,
              bbox_to_anchor=(0.5, 1.13), fontsize=7, columnspacing=1.0, handletextpad=0.4)
    plt.tight_layout(pad=0.3)
    R._save(fig, f'cross_model_{dataset}')


cross_model_2(stats, 'mmlu')
print("R1 figures regenerated: cost_accuracy_{mmlu,triviaqa}_phi3, edge_error_{mmlu,triviaqa}_phi3, cross_model_mmlu")
