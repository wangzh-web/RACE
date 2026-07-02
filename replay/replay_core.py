"""Replay RACE routing dynamics from recorded per-sample (uncertainty, edge_correct).

Mirrors cloud_migration/run_main_experiments.py 'reco'/'aci'/'static' methods 1:1.
Acceptance: reproduce main-table CIR=8.15% / EdgeErr=28.69% (MMLU, Phi-3).
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReplayResult:
    cir: float
    edge_err: float
    edge_total: int
    cloud_calls: int
    terminal_viol: bool
    window_viol: float  # fraction of steps where sliding-window edge error > alpha
    tau_history: List[float] = field(repr=False, default_factory=list)


def replay_reco(u: List[float], edge_correct: List[int], alpha: float = 0.30,
                tau0: float = 0.5, lr0: float = 0.1, gamma: float = 0.99,
                clip_lo: float = 0.01, clip_hi: float = 0.99,
                monitor_window: int = 100) -> ReplayResult:
    """Mirror of run_main_experiments.py 'reco' update (per-sample, decayed lr)."""
    tau, lr = tau0, lr0
    cloud_calls = edge_total = edge_errors = 0
    tau_hist, recent, win_viol_steps = [], [], 0
    n = len(u)
    for t in range(n):
        use_cloud = u[t] > tau
        if use_cloud:
            cloud_calls += 1
        else:
            edge_total += 1
            if not edge_correct[t]:
                edge_errors += 1
        # update uses edge model's own correctness, computed for every request
        if not edge_correct[t]:
            tau = tau - lr * (1 - alpha)
        else:
            tau = tau + lr * alpha
        tau = max(clip_lo, min(clip_hi, tau))
        lr = lr * gamma
        tau_hist.append(tau)
        # WindowViol monitoring (edge-routed losses only, paper Algorithm 1 semantics)
        recent.append(0 if (use_cloud or edge_correct[t]) else 1 if not use_cloud else 0)
        if len(recent) > monitor_window:
            recent.pop(0)
        if edge_total > 0 and sum(recent) / len(recent) > alpha:
            win_viol_steps += 1
    edge_err = edge_errors / edge_total if edge_total else 0.0
    return ReplayResult(cir=cloud_calls / n, edge_err=edge_err, edge_total=edge_total,
                        cloud_calls=cloud_calls, terminal_viol=edge_err > alpha,
                        window_viol=win_viol_steps / n, tau_history=tau_hist)


if __name__ == '__main__':
    for ds, paper_cir, paper_err in [('mmlu', 8.15, 28.69), ('triviaqa', 7.5, 26.6)]:
        d = json.load(open(f'results/assumption_verification/assumption_data_{ds}.json'))
        r = replay_reco(d['uncertainty_scores'], d['correctness'])
        print(f"{ds}: CIR={r.cir*100:.2f}% (paper {paper_cir}) | "
              f"EdgeErr={r.edge_err*100:.2f}% (paper {paper_err}) | "
              f"cloud_calls={r.cloud_calls} | TerminalViol={r.terminal_viol} | "
              f"WindowViol={r.window_viol*100:.1f}%")
