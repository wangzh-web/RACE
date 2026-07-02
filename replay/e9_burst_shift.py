"""E9: burst-shift stress test for the risk-budget envelope.

Construction: real MMLU per-sample records, subjects ranked by edge error
rate. Stream = easy-subject prefix (length B) followed by a hard-subject
burst, arriving LATE in the stream when the P-OGD step size has decayed
to ~0 -- the regime the envelope is designed for.

Compared on identical streams:
  envelope    -- canonical Algorithm 1 (0.8/1.2 budget envelope)
  no-envelope -- identical but use_budget=False (pure P-OGD)

Metrics: peak sliding-window edge error during the burst, window-violation
fraction within the burst, CIR inside the burst (responsiveness), terminal
EdgeErr / violation.
"""
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 6789]
RD = Path(__file__).resolve().parent.parent.parent / "gpu_package" / "results"


def replay_traced(u: List[float], correct: List[bool], alpha: float = 0.30,
                  tau0: float = 0.5, eta0: float = 0.1, rho: float = 0.99,
                  window: int = 100, use_budget: bool = True,
                  burst_start: int = 0) -> Dict:
    n = len(u)
    tau, eta = tau0, eta0
    b_rem = alpha * n
    cloud = edge_total = edge_err = 0
    burst_cloud = burst_n = burst_winviol = 0
    peak_win_err = 0.0
    win: List[int] = []
    for t in range(n):
        if use_budget:
            b_t = b_rem / (n - t)
            tau_adj = 0.8 * tau if b_t < alpha / 2 else (
                min(0.99, 1.2 * tau) if b_t > 1.5 * alpha else tau)
        else:
            tau_adj = tau
        is_burst = t >= burst_start
        if is_burst:
            burst_n += 1
        if u[t] > tau_adj:
            cloud += 1
            if is_burst:
                burst_cloud += 1
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
            if is_burst:
                peak_win_err = max(peak_win_err, e_hat)
                if e_hat > alpha:
                    burst_winviol += 1
            tau = max(0.01, min(0.99, tau - eta * (e_hat - alpha)))
        eta *= rho
    e = edge_err / edge_total if edge_total else 0.0
    return {
        "cir": cloud / n, "edge_err": e, "viol": e > alpha,
        "burst_cir": burst_cloud / burst_n if burst_n else 0.0,
        "burst_winviol": burst_winviol / burst_n if burst_n else 0.0,
        "peak_win_err": peak_win_err,
    }


def build_stream(recs: List[Dict], seed: int, prefix_len: int = 1600,
                 burst_len: int = 400) -> Tuple[List[float], List[bool]]:
    """Easy-subject prefix then hard-subject burst, sampled with replacement-free draws."""
    by_subj: Dict[str, List[Dict]] = defaultdict(list)
    for r in recs:
        by_subj[r["subject"]].append(r)
    err_rate = {s: 1 - sum(x["edge_correct"] for x in v) / len(v)
                for s, v in by_subj.items()}
    ranked = sorted(by_subj, key=lambda s: err_rate[s])
    half = len(ranked) // 2
    easy = [r for s in ranked[:half] for r in by_subj[s]]
    hard = [r for s in ranked[half:] for r in by_subj[s]]
    rng = random.Random(seed)
    prefix = rng.sample(easy, min(prefix_len, len(easy)))
    burst = rng.sample(hard, min(burst_len, len(hard)))
    stream = prefix + burst
    return ([r["u_sep_mmlu"] for r in stream],
            [r["edge_correct"] for r in stream])


def main() -> None:
    recs = [json.loads(l) for l in open(RD / "persample_phi3_mmlu_test.jsonl")]
    rows = {"envelope": [], "no-envelope": []}
    for seed in SEEDS:
        u, c = build_stream(recs, seed)
        burst_start = len(u) - 400
        rows["envelope"].append(replay_traced(u, c, use_budget=True,
                                              burst_start=burst_start))
        rows["no-envelope"].append(replay_traced(u, c, use_budget=False,
                                                 burst_start=burst_start))
    print(f"E9 burst-shift (easy prefix 1600 -> hard burst 400, {len(SEEDS)} seeds)")
    print(f"{'config':<12} {'EdgeErr':>8} {'Viol':>6} {'PeakWinErr':>11} "
          f"{'BurstWinViol':>13} {'BurstCIR':>9} {'CIR':>7}")
    summary = {}
    for k, v in rows.items():
        mean = lambda f: statistics.fmean(x[f] for x in v)
        viols = sum(x["viol"] for x in v)
        print(f"{k:<12} {mean('edge_err')*100:7.2f}% {viols:>3}/10 "
              f"{mean('peak_win_err')*100:10.1f}% {mean('burst_winviol')*100:12.1f}% "
              f"{mean('burst_cir')*100:8.1f}% {mean('cir')*100:6.1f}%")
        summary[k] = {f: mean(f) for f in
                      ("cir", "edge_err", "burst_cir", "burst_winviol", "peak_win_err")}
        summary[k]["viol_count"] = viols
    out = Path(__file__).parent / "out_e9_burst.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
