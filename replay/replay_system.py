"""System-accuracy replay: fold cloud GPT-5 answers into the canonical routing.

replay_compare.py validated the routing dynamics (CIR / EdgeErr) but had no cloud
answers, so it could not report end-to-end *system* accuracy. This module replays
the SAME canonical Algorithm-1 routing and additionally threads per-question cloud
correctness so it can report:

  edge_acc      -- All-Edge baseline (route nothing to cloud)
  cloud_acc     -- All-Cloud baseline (route everything to cloud)
  race_sys_acc  -- RACE: cloud_correct on cloud-routed, edge_correct on edge-kept
  P_sys         -- system reliability = race_sys_acc (= 1 - system error rate)
  CIR / EdgeErr -- reproduced from replay_compare for cross-check (drift guard)

Alignment: persample JSONL is in streaming order (`i`); cloud_gpt5_{ds}.jsonl is
keyed by global_idx. We align cloud_correct to persample order via global_idx.

Usage:
  python replay_system.py --dataset mmlu       # uses u_sep_mmlu (in-domain)
  python replay_system.py --dataset triviaqa   # uses u_sep_triviaqa
  python replay_system.py --dataset mmlu --alpha 0.35
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import replay_compare as rc

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
PERSAMPLE = {
    "mmlu": ROOT / "gpu_package" / "results" / "persample_phi3_mmlu_test.jsonl",
    "triviaqa": ROOT / "gpu_package" / "results" / "persample_phi3_triviaqa_test.jsonl",
    "mistral_mmlu": ROOT / "gpu_package" / "results" / "persample_mistral_mmlu_test.jsonl",
}
CLOUD = {
    "mmlu": BASE / "cloud_gpt5_mmlu.jsonl",
    "triviaqa": BASE / "cloud_gpt5_triviaqa.jsonl",
    "mistral_mmlu": BASE / "cloud_gpt5_mmlu.jsonl",   # same MMLU questions -> reuse cloud answers
}
SCORE_KEY = {"mmlu": "u_sep_mmlu", "triviaqa": "u_sep_triviaqa", "mistral_mmlu": "u_sep_mmlu"}


@dataclass
class SysResult:
    cir: float
    edge_err: float
    edge_acc: float
    cloud_acc: float
    race_sys_acc: float
    cloud_calls: int
    terminal_viol: bool
    n: int


def load_aligned(dataset: str) -> Tuple[List[float], List[bool], List[bool]]:
    """Return (u, edge_correct, cloud_correct) in persample streaming order."""
    cloud_by_gidx: Dict[int, bool] = {}
    for line in open(CLOUD[dataset]):
        r = json.loads(line)
        cloud_by_gidx[r["global_idx"]] = bool(r["cloud_correct"])
    u, edge_correct, cloud_correct = [], [], []
    missing = 0
    for line in open(PERSAMPLE[dataset]):
        r = json.loads(line)
        gidx = r["global_idx"]
        if gidx not in cloud_by_gidx:
            missing += 1
            continue
        u.append(float(r[SCORE_KEY[dataset]]))
        edge_correct.append(bool(r["edge_correct"]))
        cloud_correct.append(cloud_by_gidx[gidx])
    if missing:
        print(f"[warn] {missing} persample rows have no cloud answer yet (run still in progress?)")
    return u, edge_correct, cloud_correct


def replay_canonical_sys(u: List[float], edge_correct: List[bool],
                         cloud_correct: List[bool], alpha: float = 0.30,
                         tau0: float = 0.5, eta0: float = 0.1, rho: float = 0.99,
                         window: int = 100, c_lo: float = 0.8, c_hi: float = 1.2,
                         use_budget: bool = True) -> SysResult:
    """Faithful copy of replay_compare.replay_canonical + system-accuracy thread.

    Routing math is byte-identical to replay_compare.replay_canonical; the only
    additions are the cloud_correct accumulation and baseline accuracies.
    """
    n = len(u)
    tau, eta = tau0, eta0
    b_rem = alpha * n
    cloud = edge_total = edge_err = 0
    sys_correct = 0
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
            sys_correct += int(cloud_correct[t])
        else:
            edge_total += 1
            sys_correct += int(edge_correct[t])
            if not edge_correct[t]:
                edge_err += 1
                b_rem -= 1
            win.append(0 if edge_correct[t] else 1)
            if len(win) > window:
                win.pop(0)
        if win:
            e_hat = sum(win) / len(win)
            tau = max(0.01, min(0.99, tau - eta * (e_hat - alpha)))
        eta *= rho
    e = edge_err / edge_total if edge_total else 0.0
    return SysResult(
        cir=cloud / n, edge_err=e,
        edge_acc=sum(edge_correct) / n, cloud_acc=sum(cloud_correct) / n,
        race_sys_acc=sys_correct / n, cloud_calls=cloud,
        terminal_viol=e > alpha, n=n,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["mmlu", "triviaqa", "mistral_mmlu"])
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--rho", type=float, default=0.99)
    args = ap.parse_args()

    u, edge_correct, cloud_correct = load_aligned(args.dataset)
    res = replay_canonical_sys(u, edge_correct, cloud_correct, alpha=args.alpha, rho=args.rho)

    # Drift guard: routing metrics must match replay_compare.replay_canonical exactly.
    ref = rc.replay_canonical(u, edge_correct, alpha=args.alpha, rho=args.rho)
    assert abs(ref.cir - res.cir) < 1e-12 and abs(ref.edge_err - res.edge_err) < 1e-12, \
        f"DRIFT: replay_system routing != replay_compare ({ref.cir} vs {res.cir})"

    print(f"=== {args.dataset}  alpha={args.alpha}  n={res.n} ===")
    print(f"CIR          = {res.cir*100:6.2f}%   (cloud_calls={res.cloud_calls})")
    print(f"EdgeErr      = {res.edge_err*100:6.2f}%   TerminalViol={'Y' if res.terminal_viol else 'N'}")
    print(f"--- accuracies ---")
    print(f"All-Edge acc = {res.edge_acc*100:6.2f}%")
    print(f"All-Cloud acc= {res.cloud_acc*100:6.2f}%")
    print(f"RACE sys acc = {res.race_sys_acc*100:6.2f}%   (P_sys; cloud on routed, edge on kept)")
    print(f"[drift-guard] routing matches replay_compare.replay_canonical ✓")


if __name__ == "__main__":
    main()
