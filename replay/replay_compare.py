"""Dual-implementation replay: legacy (published numbers) vs canonical Algorithm 1.

Consumes the per-sample JSONL produced by gpu_package/collect_per_sample.py and
replays both routing implementations over identical inputs:

  legacy    -- cloud_migration/run_main_experiments.py 'reco': per-sample update
               tau -= lr*(1-a) on error / tau += lr*a on correct, lr *= 0.99.
               Acceptance target: MMLU CIR=8.15%, EdgeErr=28.69% (cloud_calls=163).

  canonical -- paper Algorithm 1: sliding-window P-OGD with the stated sign
               (tau_{t+1} = Proj[tau_t - eta_t*(e_hat - alpha)]), multiplicative
               eta decay, and the 0.8/1.2 risk-budget envelope. Window collects
               losses of edge-routed requests only (the paper's online-calibration section).

Usage:
  python replay_compare.py --records gpu_package/results/persample_phi3_mmlu_test.jsonl \
                           --score-key u_sep_mmlu [--alpha 0.30]
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Metrics:
    name: str
    cir: float
    edge_err: float
    cloud_calls: int
    terminal_viol: bool
    window_viol: float
    final_tau: float

    def row(self) -> str:
        return (f"{self.name:<10} CIR={self.cir*100:6.2f}%  EdgeErr={self.edge_err*100:6.2f}%  "
                f"cloud={self.cloud_calls:4d}  TermViol={'Y' if self.terminal_viol else 'N'}  "
                f"WindowViol={self.window_viol*100:5.1f}%  finalTau={self.final_tau:.3f}")


def _finalize(name: str, n: int, cloud: int, edge_total: int, edge_err: int,
              alpha: float, win_viol_steps: int, tau: float) -> Metrics:
    e = edge_err / edge_total if edge_total else 0.0
    return Metrics(name=name, cir=cloud / n, edge_err=e, cloud_calls=cloud,
                   terminal_viol=e > alpha, window_viol=win_viol_steps / n,
                   final_tau=tau)


def replay_legacy(u: List[float], correct: List[bool], alpha: float = 0.30,
                  tau0: float = 0.5, lr0: float = 0.1, rho: float = 0.99,
                  monitor_w: int = 100) -> Metrics:
    """Mirror of run_main_experiments.py 'reco' (produced the published numbers)."""
    tau, lr = tau0, lr0
    cloud = edge_total = edge_err = win_viol = 0
    win: List[int] = []
    for t in range(len(u)):
        if u[t] > tau:
            cloud += 1
        else:
            edge_total += 1
            if not correct[t]:
                edge_err += 1
            win.append(0 if correct[t] else 1)
            if len(win) > monitor_w:
                win.pop(0)
        if not correct[t]:
            tau -= lr * (1 - alpha)
        else:
            tau += lr * alpha
        tau = max(0.01, min(0.99, tau))
        lr *= rho
        if win and sum(win) / len(win) > alpha:
            win_viol += 1
    return _finalize("legacy", len(u), cloud, edge_total, edge_err, alpha,
                     win_viol, tau)


def replay_canonical(u: List[float], correct: List[bool], alpha: float = 0.30,
                     tau0: float = 0.5, eta0: float = 0.1, rho: float = 0.99,
                     window: int = 100, c_lo: float = 0.8, c_hi: float = 1.2,
                     use_budget: bool = True) -> Metrics:
    """Paper Algorithm 1: windowed P-OGD (stated sign) + risk-budget envelope."""
    n = len(u)
    tau, eta = tau0, eta0
    b_rem = alpha * n
    cloud = edge_total = edge_err = win_viol = 0
    win: List[int] = []
    for t in range(n):
        if use_budget:
            b_t = b_rem / (n - t)
            if b_t < alpha / 2:
                tau_adj = c_lo * tau
            elif b_t > 1.5 * alpha:
                tau_adj = min(0.99, c_hi * tau)
            else:
                tau_adj = tau
        else:
            tau_adj = tau
        if u[t] > tau_adj:
            cloud += 1
        else:
            edge_total += 1
            if not correct[t]:
                edge_err += 1
                b_rem -= 1
            win.append(0 if correct[t] else 1)
            if len(win) > window:
                win.pop(0)
        if win:
            e_hat = sum(win) / len(win)
            tau = max(0.01, min(0.99, tau - eta * (e_hat - alpha)))
        eta *= rho
        if win and sum(win) / len(win) > alpha:
            win_viol += 1
    return _finalize("canonical", n, cloud, edge_total, edge_err, alpha,
                     win_viol, tau)


def load_records(path: Path, score_key: str):
    u, correct = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            u.append(float(r[score_key]))
            correct.append(bool(r["edge_correct"]))
    return u, correct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--score-key", default="u_sep_mmlu")
    ap.add_argument("--alpha", type=float, default=0.30)
    args = ap.parse_args()

    u, correct = load_records(Path(args.records), args.score_key)
    edge_acc = sum(correct) / len(correct)
    print(f"records={len(u)}  edge_base_acc={edge_acc*100:.1f}%  "
          f"score[{args.score_key}] med={sorted(u)[len(u)//2]:.4f}")
    print(replay_legacy(u, correct, alpha=args.alpha).row())
    print(replay_canonical(u, correct, alpha=args.alpha).row())
    norb = replay_canonical(u, correct, alpha=args.alpha, use_budget=False)
    norb.name = "canon-NoRB"
    print(norb.row())


if __name__ == "__main__":
    main()
