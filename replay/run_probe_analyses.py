"""E1/E2/E5: probe-level analyses over regenerated per-sample records.

  E1: cross-dataset SEP 2x2 — AUROC of each SEP checkpoint on each
             dataset + reliability (monotonicity) bins + end-to-end RACE replay
             with cross-domain scores
  E2: deployment-temperature sensitivity — AUROC and RACE replay where
             routing uses the (temperature-independent) SEP score but edge
             correctness comes from responses sampled at each temperature
  E5: SEP + MaxProb fusion — logistic fusion trained on calibration,
             AUROC on test; plus each signal alone

Dependency-light: AUROC computed rank-based; logistic regression via plain
gradient descent (no sklearn required).

Usage:
  python run_probe_analyses.py --results-dir gpu_package/results --out analysis/replay/out_probe.json
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from replay_compare import replay_canonical


def auroc(scores: Sequence[float], is_error: Sequence[bool]) -> float:
    """Rank-based AUROC: P(score_error > score_correct), ties = 0.5."""
    pairs = sorted(zip(scores, is_error))
    n_pos = sum(is_error)
    n_neg = len(is_error) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average rank for the tie group
        rank_sum += avg_rank * sum(1 for k in range(i, j) if pairs[k][1])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def reliability_bins(scores: List[float], is_error: List[bool],
                     n_bins: int = 10) -> List[Dict]:
    """Quantile bins -> error rate per bin (monotonicity check, Assumption 2)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    bins = []
    size = len(order) // n_bins
    for b in range(n_bins):
        idx = order[b * size:(b + 1) * size] if b < n_bins - 1 else order[b * size:]
        errs = [is_error[i] for i in idx]
        bins.append({"bin": b, "mean_score": sum(scores[i] for i in idx) / len(idx),
                     "error_rate": sum(errs) / len(errs), "n": len(idx)})
    return bins


def is_monotone(bins: List[Dict], tol: float = 0.02) -> bool:
    rates = [b["error_rate"] for b in bins]
    return all(rates[i + 1] >= rates[i] - tol for i in range(len(rates) - 1))


def logistic_fit(x: List[List[float]], y: List[bool], epochs: int = 400,
                 lr: float = 0.5) -> List[float]:
    """Plain-GD logistic regression with intercept; features standardized."""
    d = len(x[0])
    mu = [sum(r[j] for r in x) / len(x) for j in range(d)]
    sd = [max(1e-8, math.sqrt(sum((r[j] - mu[j]) ** 2 for r in x) / len(x)))
          for j in range(d)]
    xs = [[(r[j] - mu[j]) / sd[j] for j in range(d)] for r in x]
    w = [0.0] * (d + 1)
    for _ in range(epochs):
        g = [0.0] * (d + 1)
        for xi, yi in zip(xs, y):
            z = w[0] + sum(w[j + 1] * xi[j] for j in range(d))
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = p - (1.0 if yi else 0.0)
            g[0] += e
            for j in range(d):
                g[j + 1] += e * xi[j]
        for j in range(d + 1):
            w[j] -= lr * g[j] / len(xs)
    return _unstandardize(w, mu, sd)


def _unstandardize(w: List[float], mu: List[float], sd: List[float]) -> List[float]:
    d = len(mu)
    raw = [w[j + 1] / sd[j] for j in range(d)]
    b0 = w[0] - sum(raw[j] * mu[j] for j in range(d))
    return [b0, *raw]


def logistic_score(w: List[float], row: List[float]) -> float:
    z = w[0] + sum(w[j + 1] * row[j] for j in range(len(row)))
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def load(path: Path) -> List[Dict]:
    return [json.loads(l) for l in open(path)]


def e1_cross_dataset(rd: Path, alpha: float) -> Dict:
    out: Dict = {"auroc": {}, "reliability": {}, "replay": {}}
    for ds in ("mmlu", "triviaqa"):
        recs = load(rd / f"persample_phi3_{ds}_test.jsonl")
        errs = [not r["edge_correct"] for r in recs]
        for sep in ("mmlu", "triviaqa"):
            key = f"u_sep_{sep}"
            scores = [r[key] for r in recs]
            tag = f"sep[{sep}]->data[{ds}]"
            out["auroc"][tag] = round(auroc(scores, errs), 4)
            bins = reliability_bins(scores, errs)
            out["reliability"][tag] = {"bins": bins, "monotone": is_monotone(bins)}
            m = replay_canonical(scores, [not e for e in errs], alpha=alpha)
            out["replay"][tag] = {"cir": m.cir, "edge_err": m.edge_err,
                                  "viol": m.terminal_viol,
                                  "window_viol": m.window_viol}
    return out


def e2_temperature(rd: Path, alpha: float) -> Dict:
    base = load(rd / "persample_phi3_mmlu_test.jsonl")
    u = [r["u_sep_mmlu"] for r in base]
    out: Dict = {}
    variants = {"greedy(T=0)": [r["edge_correct"] for r in base]}
    for t in ("0.3", "1.0", "1.5"):
        p = rd / f"temperature_phi3_mmlu_T{t}.jsonl"
        if p.exists():
            recs = load(p)
            variants[f"T={t}"] = [r["edge_correct"] for r in recs]
    for name, correct in variants.items():
        n = min(len(u), len(correct))
        errs = [not c for c in correct[:n]]
        m = replay_canonical(u[:n], correct[:n], alpha=alpha)
        out[name] = {"edge_base_acc": 1 - sum(errs) / n,
                     "auroc": round(auroc(u[:n], errs), 4),
                     "cir": m.cir, "edge_err": m.edge_err,
                     "viol": m.terminal_viol, "window_viol": m.window_viol}
    return out


def e5_fusion(rd: Path) -> Dict:
    out: Dict = {}
    for ds in ("mmlu", "triviaqa"):
        cal = load(rd / f"persample_phi3_{ds}_cal.jsonl")
        test = load(rd / f"persample_phi3_{ds}_test.jsonl")
        key = f"u_sep_{ds}"
        # uncertainty-style features: SEP score and (1 - maxprob)
        xc = [[r[key], 1 - r["maxprob"]] for r in cal]
        yc = [not r["edge_correct"] for r in cal]
        w = logistic_fit(xc, yc)
        xt = [[r[key], 1 - r["maxprob"]] for r in test]
        yt = [not r["edge_correct"] for r in test]
        out[ds] = {
            "auroc_sep": round(auroc([r[key] for r in test], yt), 4),
            "auroc_maxprob": round(auroc([1 - r["maxprob"] for r in test], yt), 4),
            "auroc_fusion": round(auroc([logistic_score(w, x) for x in xt], yt), 4),
            "weights": [round(v, 4) for v in w],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rd = Path(args.results_dir)

    results = {
        "E1_cross_dataset": e1_cross_dataset(rd, args.alpha),
        "E2_temperature": e2_temperature(rd, args.alpha),
        "E5_fusion": e5_fusion(rd),
    }
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {args.out}")
    print("E1 AUROC:", results["E1_cross_dataset"]["auroc"])
    print("E5:", {k: {kk: vv for kk, vv in v.items() if kk.startswith('auroc')}
                  for k, v in results["E5_fusion"].items()})


if __name__ == "__main__":
    main()
