"""Canonical replay of all main-table baselines over recorded per-sample signals.

Routing logic ported verbatim from:
  - cloud_migration/run_main_experiments.py : all_edge, static (Static-CP), aci (ACI)
  - cloud_migration/run_baselines.py        : random_0.5, latency_first, frugalgpt, larc

No GPU / no model inference: each method is a deterministic (or seeded) function of
the recorded SEP uncertainty (u_sep_*), edge_correct, and (L-ARC) hidden states.
System accuracy folds in the cached GPT-5 cloud answers (cloud_gpt5_{ds}.jsonl).

CIR / EdgeErr are cloud-model-independent and serve as a VALIDATION GATE: they must
reproduce the published main-table values. Acc shifts (GPT-4 -> GPT-5 cloud).

Usage:
  python replay_baselines.py --dataset mmlu
  python replay_baselines.py --dataset triviaqa
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
PERSAMPLE = {
    "mmlu": ROOT / "gpu_package/results/persample_phi3_mmlu_test.jsonl",
    "triviaqa": ROOT / "gpu_package/results/persample_phi3_triviaqa_test.jsonl",
}
HIDDEN = {
    "mmlu": ROOT / "gpu_package/results/hidden_phi3_mmlu_test.npy",
    "triviaqa": ROOT / "gpu_package/results/hidden_phi3_triviaqa_test.npy",
}
CLOUD = {
    "mmlu": BASE / "cloud_gpt5_mmlu.jsonl",
    "triviaqa": BASE / "cloud_gpt5_triviaqa.jsonl",
}
SCORE = {"mmlu": "u_sep_mmlu", "triviaqa": "u_sep_triviaqa"}
ALPHA = 0.30
SEEDS = [42, 123, 456]

# Published main-table targets (CIR, EdgeErr) for the validation gate.
PUBLISHED = {
    "mmlu": {
        "all_cloud": (100.0, 0.0), "all_edge": (0.0, 30.9), "random_0.5": (49.6, 28.6),
        "latency_first": (0.0, 28.6), "static": (15.0, 27.0), "aci": (18.4, 24.9),
        "frugalgpt": (35.0, 14.2), "larc": (27.2, 17.8),
    },
    "triviaqa": {
        "all_cloud": (100.0, 0.0), "all_edge": (0.0, 29.9), "random_0.5": (49.6, 25.6),
        "latency_first": (0.0, 26.3), "static": (10.2, 25.8), "aci": (11.1, 25.4),
        "frugalgpt": (26.3, 21.2), "larc": (5.4, 25.9),
    },
}


def load(dataset: str):
    rows = [json.loads(l) for l in open(PERSAMPLE[dataset])]
    u = np.array([float(r[SCORE[dataset]]) for r in rows])
    edge_ok = np.array([bool(r["edge_correct"]) for r in rows])
    gidx = [r["global_idx"] for r in rows]
    cloud_by_g = {json.loads(l)["global_idx"]: bool(json.loads(l)["cloud_correct"])
                  for l in open(CLOUD[dataset])}
    cloud_ok = np.array([cloud_by_g[g] for g in gidx])
    hidden = np.load(HIDDEN[dataset])
    return u, edge_ok, cloud_ok, hidden


def metrics(route_cloud: np.ndarray, edge_ok: np.ndarray, cloud_ok: np.ndarray):
    n = len(route_cloud)
    cir = route_cloud.mean()
    kept = ~route_cloud
    edge_err = (kept & ~edge_ok).sum() / kept.sum() if kept.sum() else 0.0
    sys_ok = np.where(route_cloud, cloud_ok, edge_ok)
    return cir * 100, edge_err * 100, sys_ok.mean() * 100, int(route_cloud.sum())


# ---- routing functions (return boolean route-to-cloud mask) ----

def route_all_edge(u, edge_ok, hidden): return np.zeros(len(u), bool)
def route_static(u, edge_ok, hidden): return u > 0.5


def route_aci(u, edge_ok, hidden, alpha=ALPHA, tau0=0.5, lr=0.1):
    tau = tau0
    out = np.zeros(len(u), bool)
    for t in range(len(u)):
        out[t] = u[t] > tau
        if not edge_ok[t]:
            tau -= lr * (1 - alpha)
        else:
            tau += lr * alpha
        tau = max(0.01, min(0.99, tau))
    return out


def route_frugalgpt(u, edge_ok, hidden, thr0=0.3):
    thr = thr0
    out = np.zeros(len(u), bool)
    for t in range(len(u)):
        quality = 1.0 - u[t]
        use_cloud = quality < thr
        out[t] = use_cloud
        if not use_cloud:  # update only when edge used (port from run_baselines.py)
            if not edge_ok[t]:
                thr = min(0.9, thr + 0.05)
            else:
                thr = max(0.1, thr - 0.01)
    return out


def _larc_local_tau(h, svs, ws, global_tau, bandwidth=1.0):
    if not svs:
        return global_tau
    ksum = tw = 0.0
    for sv, w in zip(svs, ws):
        k = np.exp(-np.sum((h - sv) ** 2) / (2 * bandwidth ** 2))
        ksum += w * k
        tw += k
    if tw < 1e-6:
        return global_tau
    return 1.0 / (1.0 + np.exp(-(ksum / tw)))


def route_larc(u, edge_ok, hidden, alpha=ALPHA, tau0=0.5, lr=0.1, max_sv=100, gamma=0.1):
    tau = tau0
    svs: List[np.ndarray] = []
    ws: List[float] = []
    out = np.zeros(len(u), bool)
    for t in range(len(u)):
        local_tau = _larc_local_tau(hidden[t], svs, ws, tau)
        use_cloud = u[t] > local_tau
        out[t] = use_cloud
        if not use_cloud:
            grad = (0 if edge_ok[t] else 1) - alpha
            if len(svs) < max_sv:
                svs.append(hidden[t].copy())
                ws.append(grad)
            else:
                d = [np.sum((sv - hidden[t]) ** 2) for sv in svs]
                ws[int(np.argmin(d))] += gamma * grad
            if not edge_ok[t]:
                tau -= lr * (1 - alpha)
            else:
                tau += lr * alpha
            tau = max(0.01, min(0.99, tau))
    return out


def route_random(u, edge_ok, hidden, p=0.5, seed=42):
    rng = np.random.RandomState(seed)
    return rng.random(len(u)) < p


DETERMINISTIC = {
    "all_edge": route_all_edge, "latency_first": route_all_edge,  # degenerates to all-edge
    "static": route_static, "aci": route_aci,
    "frugalgpt": route_frugalgpt, "larc": route_larc,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mmlu", "triviaqa"])
    args = ap.parse_args()
    u, edge_ok, cloud_ok, hidden = load(args.dataset)
    pub = PUBLISHED[args.dataset]

    print(f"=== {args.dataset}  n={len(u)}  (GPT-5 cloud)  [validation gate: CIR/EdgeErr vs published] ===")
    print(f"{'method':<14}{'CIR':>8}{'EdgeErr':>9}{'Acc(GPT5)':>11}   {'pub CIR/Err':>14}  {'gate':>5}")

    # All-Cloud / All-Edge reference rows
    n = len(u)
    allcloud_acc = cloud_ok.mean() * 100
    print(f"{'all_cloud':<14}{100.0:>7.1f}%{0.0:>8.1f}%{allcloud_acc:>10.2f}%   {'100.0/0.0':>14}  {'ref':>5}")

    rows_out = {"all_cloud": {"cir": 100.0, "edge_err": 0.0, "acc": round(allcloud_acc, 2)}}

    for name, fn in DETERMINISTIC.items():
        mask = fn(u, edge_ok, hidden)
        cir, err, acc, calls = metrics(mask, edge_ok, cloud_ok)
        pc, pe = pub[name]
        gate = "OK" if abs(cir - pc) <= 1.0 and abs(err - pe) <= 1.5 else "CHECK"
        print(f"{name:<14}{cir:>7.2f}%{err:>8.2f}%{acc:>10.2f}%   {f'{pc}/{pe}':>14}  {gate:>5}")
        rows_out[name] = {"cir": round(cir, 2), "edge_err": round(err, 2), "acc": round(acc, 2)}

    # Random: average over seeds
    accs, cirs, errs = [], [], []
    for s in SEEDS:
        mask = route_random(u, edge_ok, hidden, p=0.5, seed=s)
        c, e, a, _ = metrics(mask, edge_ok, cloud_ok)
        cirs.append(c); errs.append(e); accs.append(a)
    pc, pe = pub["random_0.5"]
    gate = "OK" if abs(np.mean(cirs) - pc) <= 2.0 else "CHECK"
    print(f"{'random_0.5':<14}{np.mean(cirs):>7.2f}%{np.mean(errs):>8.2f}%{np.mean(accs):>10.2f}%   "
          f"{f'{pc}/{pe}':>14}  {gate:>5}  (±{np.std(accs):.2f} acc over {len(SEEDS)} seeds)")
    rows_out["random_0.5"] = {"cir": round(float(np.mean(cirs)), 2),
                              "edge_err": round(float(np.mean(errs)), 2),
                              "acc": round(float(np.mean(accs)), 2)}

    out = BASE / f"baselines_gpt5_{args.dataset}.json"
    json.dump(rows_out, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
