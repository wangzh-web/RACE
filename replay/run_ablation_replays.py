"""E3/E4/E7/E8: full ablation suite replayed over regenerated per-sample records.

All experiments replay the canonical Algorithm 1 (replay_compare.replay_canonical)
over identical recorded inputs, so results are deterministic and exactly
reproducible from the released logs.

  E3: conservative/aggressive factor grid c_lo x c_hi
  E4: online vs static / oracle-static / random thresholds,
             on the natural (subject-grouped, shifted) stream and on
             shuffled (stationary) streams over 10 seeds
  E7: alpha sweep with CIR / EdgeErr / TerminalViol / WindowViol
  E8: decay factor rho sweep

Usage:
  python run_ablation_replays.py \
      --records gpu_package/results/persample_phi3_mmlu_test.jsonl \
      --cal gpu_package/results/persample_phi3_mmlu_cal.jsonl \
      --score-key u_sep_mmlu --out analysis/replay/out_mmlu
"""
import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from replay_compare import Metrics, replay_canonical, load_records

ALPHA_DEFAULT = 0.30
SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 6789]


def static_threshold_replay(u: List[float], correct: List[bool], tau: float,
                            alpha: float) -> Metrics:
    n = len(u)
    cloud = edge_total = edge_err = win_viol = 0
    win: List[int] = []
    for t in range(n):
        if u[t] > tau:
            cloud += 1
        else:
            edge_total += 1
            if not correct[t]:
                edge_err += 1
            win.append(0 if correct[t] else 1)
            if len(win) > 100:
                win.pop(0)
        if win and sum(win) / len(win) > alpha:
            win_viol += 1
    e = edge_err / edge_total if edge_total else 0.0
    return Metrics(name=f"static@{tau:.3f}", cir=cloud / n, edge_err=e,
                   cloud_calls=cloud, terminal_viol=e > alpha,
                   window_viol=win_viol / n, final_tau=tau)


def calibrated_static_tau(cal_u: List[float], cal_correct: List[bool],
                          alpha: float) -> float:
    """Largest threshold whose calibration-set conditional edge error <= alpha."""
    order = sorted(set(cal_u))
    best = order[0]
    for tau in order:
        kept = [(u <= tau, c) for u, c in zip(cal_u, cal_correct)]
        edge = [c for k, c in kept if k]
        if edge and (1 - sum(edge) / len(edge)) <= alpha:
            best = tau
    return best


def oracle_static_tau(u: List[float], correct: List[bool], alpha: float) -> float:
    """Best fixed threshold in hindsight on the test stream itself."""
    return calibrated_static_tau(u, correct, alpha)


def e3_factor_grid(u, correct, alpha) -> List[Dict]:
    rows = []
    for c_lo in (0.7, 0.8, 0.9):
        for c_hi in (1.1, 1.2, 1.3):
            m = replay_canonical(u, correct, alpha=alpha, c_lo=c_lo, c_hi=c_hi)
            rows.append({"c_lo": c_lo, "c_hi": c_hi, "cir": m.cir,
                         "edge_err": m.edge_err, "viol": m.terminal_viol,
                         "window_viol": m.window_viol})
    return rows


def e4_online_vs_static(u, correct, cal_u, cal_correct, alpha) -> Dict:
    out: Dict = {"shifted": {}, "stationary": {}}
    # natural order = subject-grouped shift stream
    out["shifted"]["RACE(online)"] = replay_canonical(u, correct, alpha=alpha).__dict__
    t_cal = calibrated_static_tau(cal_u, cal_correct, alpha)
    out["shifted"][f"static-cal(tau={t_cal:.3f})"] = \
        static_threshold_replay(u, correct, t_cal, alpha).__dict__
    t_orc = oracle_static_tau(u, correct, alpha)
    out["shifted"][f"oracle-static(tau={t_orc:.3f})"] = \
        static_threshold_replay(u, correct, t_orc, alpha).__dict__

    # stationary = shuffled stream, mean over seeds
    agg: Dict[str, List[Metrics]] = {"RACE(online)": [], "static-cal": []}
    for seed in SEEDS:
        idx = list(range(len(u)))
        random.Random(seed).shuffle(idx)
        su = [u[i] for i in idx]
        sc = [correct[i] for i in idx]
        agg["RACE(online)"].append(replay_canonical(su, sc, alpha=alpha))
        agg["static-cal"].append(static_threshold_replay(su, sc, t_cal, alpha))
    for k, ms in agg.items():
        out["stationary"][k] = {
            "cir_mean": statistics.fmean(m.cir for m in ms),
            "edge_err_mean": statistics.fmean(m.edge_err for m in ms),
            "viol_any": any(m.terminal_viol for m in ms),
            "window_viol_mean": statistics.fmean(m.window_viol for m in ms),
        }
    return out


def e7_alpha_sweep(u, correct, alpha_values=(0.15, 0.20, 0.25, 0.30, 0.35, 0.40)) -> List[Dict]:
    rows = []
    for a in alpha_values:
        m = replay_canonical(u, correct, alpha=a)
        rows.append({"alpha": a, "cir": m.cir, "edge_err": m.edge_err,
                     "viol": m.terminal_viol, "window_viol": m.window_viol})
    return rows


def e8_rho_sweep(u, correct, alpha, rhos=(0.95, 0.99, 0.999)) -> List[Dict]:
    rows = []
    for r in rhos:
        m = replay_canonical(u, correct, alpha=alpha, rho=r)
        rows.append({"rho": r, "cir": m.cir, "edge_err": m.edge_err,
                     "viol": m.terminal_viol, "window_viol": m.window_viol})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--cal", required=True)
    ap.add_argument("--score-key", default="u_sep_mmlu")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    u, correct = load_records(Path(args.records), args.score_key)
    cal_u, cal_correct = load_records(Path(args.cal), args.score_key)

    results = {
        "meta": {"records": args.records, "score_key": args.score_key,
                 "alpha": args.alpha, "n": len(u),
                 "edge_base_acc": sum(correct) / len(correct)},
        "E3_factor_grid": e3_factor_grid(u, correct, args.alpha),
        "E4_online_vs_static": e4_online_vs_static(u, correct, cal_u,
                                                   cal_correct, args.alpha),
        "E7_alpha_sweep": e7_alpha_sweep(u, correct),
        "E8_rho_sweep": e8_rho_sweep(u, correct, args.alpha),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out.with_suffix('.json')}")
    for row in results["E7_alpha_sweep"]:
        print(f"  alpha={row['alpha']:.2f}  CIR={row['cir']*100:5.1f}%  "
              f"EdgeErr={row['edge_err']*100:5.1f}%  Viol={'Y' if row['viol'] else 'N'}")


if __name__ == "__main__":
    main()
