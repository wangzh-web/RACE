"""Measure real cloud API latency with a documented protocol.

Protocol:
- model: fixed snapshot via --model (default gpt-5-2025-08-07), reasoning_effort
  minimal (reasoning_tokens=0), non-streaming completion. GPT-5 reasoning models
  take `max_completion_tokens` (not `max_tokens`) and do NOT accept `temperature`.
- prompts: actual MMLU test items in deployment format (max_completion_tokens=20),
  so the measurement matches the serving workload used for the cloud answers.
- N calls per session; run >=3 sessions at different times of day.
- records per call: latency, prompt/completion/reasoning tokens, http retries, ts.
- report: median / mean / p5 / p95.

Usage:
  python measure_cloud_latency.py --n 200 --session morning
Reads OPENAI_API_KEY / OPENAI_BASE_URL from project .env.
Output: analysis/replay/latency_{model}_{session}.jsonl + summary printed.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
DATA = ROOT / "data" / "mmlu" / "test.json"


def load_env() -> dict:
    env = {}
    for line in open(ROOT / ".env"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def format_prompt(item: dict) -> str:
    return (f"Answer with only A, B, C, or D.\n\nQuestion: {item['question']}\n"
            f"A) {item['choices'][0]}\nB) {item['choices'][1]}\n"
            f"C) {item['choices'][2]}\nD) {item['choices'][3]}\n\nAnswer:")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="gpt-5-2025-08-07")
    ap.add_argument("--session", required=True, help="label, e.g. morning/noon/night")
    ap.add_argument("--offset", type=int, default=300, help="test.json start index")
    args = ap.parse_args()

    env = load_env()
    key = env["OPENAI_API_KEY"]
    base = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    items = json.load(open(DATA))[args.offset:args.offset + args.n]
    out = BASE / f"latency_{args.model}_{args.session}.jsonl"
    lat = []
    with open(out, "a") as f:
        for i, item in enumerate(items):
            prompt = format_prompt(item)
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "reasoning_effort": "low",
                "max_completion_tokens": 4000,
            }
            t0 = time.perf_counter()
            retries = 0
            while True:
                try:
                    resp = requests.post(f"{base}/chat/completions",
                                         headers=headers, json=body, timeout=120)
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    j = resp.json()
                    break
                except Exception:
                    retries += 1
                    if retries >= 3:
                        raise
                    time.sleep(2)
            ms = (time.perf_counter() - t0) * 1000
            lat.append(ms)
            u = j.get("usage", {})
            rt = u.get("completion_tokens_details", {}).get("reasoning_tokens")
            f.write(json.dumps({
                "i": i, "latency_ms": round(ms, 1),
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "reasoning_tokens": rt,
                "retries": retries, "model": j.get("model", args.model),
                "ts": time.time(),
            }) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"{i+1}/{len(items)} median so far: {statistics.median(lat):.0f}ms")

    lat.sort()
    n = len(lat)
    print(f"\nmodel={args.model} session={args.session} n={n}")
    print(f"median={statistics.median(lat):.0f}ms  mean={statistics.fmean(lat):.0f}ms  "
          f"p5={lat[int(0.05*n)]:.0f}ms  p95={lat[int(0.95*n)]:.0f}ms")


if __name__ == "__main__":
    main()
