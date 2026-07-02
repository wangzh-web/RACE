"""Generate cloud GPT-5 answers for the 2000 test questions per dataset (concurrent).

Canonical alignment with gpu_package/collect_per_sample.py:
- MMLU  : format_prompt_mmlu,  check_mmlu      (letter A-D), edge max_tokens=10
- TriviaQA: format_prompt_triviaqa, check_triviaqa (alias substring), edge max_tokens=50

Cloud model: gpt-5-2025-08-07, reasoning_effort=low (default; serving-realistic strong
cloud, ~95% MMLU vs ~81% at minimal). GPT-5 reasoning models do NOT accept temperature;
determinism is met by one cached call per question (keyed by global_idx). max_completion
_tokens is set high (4000) so reasoning tokens never truncate the visible answer.

Output: analysis/replay/cloud_gpt5_{dataset}.jsonl (resume-aware, concurrent).

Usage:
  python cloud_gpt5_answers.py --dataset mmlu     --reasoning low
  python cloud_gpt5_answers.py --dataset triviaqa --reasoning low --workers 8
"""
import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
DATA = {"mmlu": ROOT / "data/mmlu/test.json", "triviaqa": ROOT / "data/triviaqa/test.json"}
PERSAMPLE = {
    "mmlu": ROOT / "gpu_package/results/persample_phi3_mmlu_test.jsonl",
    "triviaqa": ROOT / "gpu_package/results/persample_phi3_triviaqa_test.jsonl",
}
MODEL = "gpt-5-2025-08-07"
MAX_TOK = 4000   # headroom for reasoning tokens (low ~150) + visible answer


def load_env():
    env = {}
    for line in open(ROOT / ".env"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def format_prompt_mmlu(item):
    return (f"Answer with only A, B, C, or D.\n\nQuestion: {item['question']}\n"
            f"A) {item['choices'][0]}\nB) {item['choices'][1]}\n"
            f"C) {item['choices'][2]}\nD) {item['choices'][3]}\n\nAnswer:")


def format_prompt_triviaqa(item):
    return f"Answer the following question concisely.\n\nQuestion: {item['question']}\n\nAnswer:"


def check_mmlu(response, item):
    for char in response.upper():
        if char in "ABCD":
            return (ord(char) - ord("A")) == int(item["answer"])
    return False


def check_triviaqa(response, item):
    pred = response.strip().lower()
    if not pred:
        return False  # empty-guard: avoid spurious 'pred in gt' true-positive
    for prefix in ["the answer is", "answer:", "it is"]:
        if pred.startswith(prefix):
            pred = pred[len(prefix):].strip()
    gt = item["answer"].lower().strip()
    if pred == gt or gt in pred or pred in gt:
        return True
    for alias in item.get("aliases", []):
        a = alias.lower().strip()
        if a and (pred == a or a in pred or pred in a):
            return True
    return False


FMT = {"mmlu": format_prompt_mmlu, "triviaqa": format_prompt_triviaqa}
CHECK = {"mmlu": check_mmlu, "triviaqa": check_triviaqa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mmlu", "triviaqa"])
    ap.add_argument("--reasoning", default="low", choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    env = load_env()
    key = env["OPENAI_API_KEY"]
    base = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    data = json.load(open(DATA[args.dataset]))
    rows = [json.loads(l) for l in open(PERSAMPLE[args.dataset])]
    if args.limit:
        rows = rows[:args.limit]

    out_path = BASE / f"cloud_gpt5_{args.dataset}.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            done.add(json.loads(line)["global_idx"])
    pending = [r for r in rows if r["global_idx"] not in done]
    fmt, check = FMT[args.dataset], CHECK[args.dataset]
    print(f"{args.dataset}: {len(done)} cached, {len(pending)} to fetch "
          f"(reasoning={args.reasoning}, workers={args.workers})", flush=True)

    lock = threading.Lock()
    fout = open(out_path, "a")
    counter = {"n": 0, "ok": 0}
    lat = []

    def work(row):
        gidx = row["global_idx"]
        item = data[gidx]
        body = {"model": MODEL, "messages": [{"role": "user", "content": fmt(item)}],
                "reasoning_effort": args.reasoning, "max_completion_tokens": MAX_TOK}
        retries = 0
        while True:
            try:
                t0 = time.perf_counter()
                resp = requests.post(f"{base}/chat/completions", headers=headers, json=body, timeout=180)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:150]}")
                j = resp.json()
                ms = (time.perf_counter() - t0) * 1000
                break
            except Exception:
                retries += 1
                if retries >= 5:
                    return None
                time.sleep(2 * retries)
        msg = j["choices"][0]["message"]["content"] or ""
        u = j.get("usage", {})
        ok = check(msg, item)
        rec = {
            "global_idx": gidx, "i": row["i"], "subject": row.get("subject"),
            "cloud_response": msg, "cloud_correct": ok,
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": u.get("completion_tokens_details", {}).get("reasoning_tokens"),
            "latency_ms": round(ms, 1), "retries": retries,
            "model": j.get("model", MODEL), "reasoning_effort": args.reasoning, "ts": time.time(),
        }
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            counter["n"] += 1
            counter["ok"] += int(ok)
            lat.append(ms)
            if counter["n"] % 100 == 0:
                print(f"  {counter['n']}/{len(pending)} acc={counter['ok']/counter['n']:.3f} "
                      f"med={statistics.median(lat):.0f}ms", flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed([ex.submit(work, r) for r in pending]))
    fout.close()

    allrows = [json.loads(l) for l in open(out_path)]
    acc = sum(r["cloud_correct"] for r in allrows) / len(allrows)
    empty = sum(1 for r in allrows if not r["cloud_response"].strip())
    print(f"DONE {args.dataset}: total={len(allrows)} acc={acc*100:.2f}% empty={empty} "
          f"fetched_now={counter['n']} (failed={len(pending)-counter['n']})", flush=True)


if __name__ == "__main__":
    main()
