"""Compare GPT-5 reasoning tiers on a representative MMLU sample (acc + latency)."""
import json, time, requests, statistics, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
env = {}
for line in open(ROOT / ".env"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); env[k] = v
key = env["OPENAI_API_KEY"]; base = env.get("OPENAI_BASE_URL")
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
mmlu = json.load(open(ROOT / "data/mmlu/test.json"))
idxs = list(range(300, 2300, 17))  # ~118 representative samples


def check(resp, it):
    for c in resp.upper():
        if c in "ABCD":
            return (ord(c) - ord("A")) == int(it["answer"])
    return False


def fmt(it):
    return (f"Answer with only A, B, C, or D.\n\nQuestion: {it['question']}\n"
            f"A) {it['choices'][0]}\nB) {it['choices'][1]}\n"
            f"C) {it['choices'][2]}\nD) {it['choices'][3]}\n\nAnswer:")


out = open(BASE / "tier_compare_result.txt", "w")
for tier in ["minimal", "low", "medium"]:
    cor = 0; lat = []; rtoks = []
    for n, gi in enumerate(idxs):
        it = mmlu[gi]
        body = {"model": "gpt-5-2025-08-07", "messages": [{"role": "user", "content": fmt(it)}],
                "reasoning_effort": tier,
                "max_completion_tokens": 20 if tier == "minimal" else 4000}
        t0 = time.perf_counter()
        try:
            j = requests.post(f"{base}/chat/completions", headers=H, json=body, timeout=180).json()
        except Exception as e:
            print(f"  {tier} err at {n}: {e}", flush=True); continue
        lat.append((time.perf_counter() - t0) * 1000)
        msg = j["choices"][0]["message"]["content"] or ""
        u = j.get("usage", {})
        rtoks.append(u.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0)
        if check(msg, it):
            cor += 1
        if (n + 1) % 30 == 0:
            print(f"  {tier} {n+1}/{len(idxs)} acc={cor/(n+1)*100:.1f} med={statistics.median(lat):.0f}ms", flush=True)
    line = (f"{tier:8s}: acc={cor/len(idxs)*100:.2f}%  n={len(idxs)}  "
            f"lat_median={statistics.median(lat):.0f}ms  lat_p95={sorted(lat)[int(0.95*len(lat))]:.0f}ms  "
            f"reasoning_tok_mean={statistics.mean(rtoks):.0f}")
    print(line, flush=True)
    out.write(line + "\n"); out.flush()
out.close()
print("DONE", flush=True)
