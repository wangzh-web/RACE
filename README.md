# RACE: Risk-Aware Cloud-Edge Orchestration with Online Adaptive Calibration for Trustworthy LLM Services

[![IEEE TSC](https://img.shields.io/badge/IEEE-TSC%20Submission-blue)](https://www.computer.org/csdl/journal/sc)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **RACE** is the first edge-cloud LLM orchestration framework that integrates online adaptive threshold calibration with semantic risk budgeting, providing provable sublinear regret bounds and finite-sample coverage guarantees under distribution shift.

---

## 📄 Paper

**Title**: RACE: Risk-Aware Cloud-Edge Orchestration with Online Adaptive Calibration for Trustworthy LLM Services

**Authors**: Zhihao Wang, Min Ren, Xia Liang, Gaoyong Li, Ding Ding, and Mauro Conti

**Status**: Submitted to *IEEE Transactions on Services Computing*

**Abstract**: Edge-deployed large language models (LLMs) face a critical trustworthiness challenge: constrained by model capacity, edge LLMs exhibit substantially higher hallucination rates on knowledge-intensive queries than their cloud counterparts, yet existing edge-cloud scheduling methods route requests based solely on latency or load balancing, lacking any awareness of—or guarantees on—semantic correctness. We propose RACE (Risk-Aware Cloud-Edge Orchestration), the first framework to integrate online adaptive threshold calibration with semantic risk budgeting. RACE introduces three key innovations: (1) a Semantic Entropy Probe (SEP) that predicts semantic uncertainty from hidden states in a single forward pass (<1 ms), enabling real-time hallucination detection on edge devices; (2) an online adaptive calibrator that fuses risk control with online convex optimization to dynamically adjust decision thresholds under distribution shift; and (3) a semantic risk budget grounded in Lagrangian duality theory that bounds the cumulative hallucination rate below a user-specified tolerance α.

---

## 🏗️ Framework Overview

RACE reformulates the scheduling problem as a **risk allocation** problem: the user specifies a maximum acceptable hallucination rate α as a risk budget, and the scheduler dynamically adjusts routing thresholds via online adaptive calibration to minimize cloud cost subject to this constraint.

### Key Components

| Component | Description |
|-----------|-------------|
| **Semantic Entropy Probe (SEP)** | Predicts semantic uncertainty from hidden states in a single forward pass (<1 ms), achieving 8,000× speedup over standard semantic entropy |
| **Online Adaptive Calibrator** | Fuses risk control with Projected Online Gradient Descent (P-OGD), exploiting local stationarity and monotonicity structure |
| **Semantic Risk Budget** | Grounded in Lagrangian duality theory, provides provable regret bounds and finite-sample coverage guarantees |

---

## 📊 Key Results

Validated on a real edge-cloud platform (RTX 5090 + GPT-4 API) with two edge models (Phi-3-medium, Mistral-7B) across MMLU and TriviaQA benchmarks:

- ✅ **Cloud call rate of only 7.5–8.2%** while maintaining edge error rates strictly below α = 30%
- ✅ **55% cost reduction** compared to ACI
- ✅ **76% cost reduction** compared to FrugalGPT
- ✅ **<1 ms routing latency** via SEP (8,000× faster than full semantic entropy)
- ✅ **Provable sublinear regret** O(L_φ η₀ / (1−ρ)) with finite-sample coverage guarantees
- ✅ **Statistical significance** validated across 10 repetitions (paired t-test, p < 0.05)

### Experimental Results

The `results/` directory contains the complete experimental data (JSON format) for all reported results, including:

- `results/cloud_10seed/` — Full 10-seed experiment results for Phi-3-medium on MMLU and TriviaQA across all methods (RACE, ACI, Static-CP, All-Edge)
- `results/paper/` — Aggregated main results table data
- `results/robustness/` — SEP robustness analysis (OOD detection, sensitivity, saliency, domain transfer)
- `results/beta_sensitivity/` — Verification cost β sensitivity analysis
- `results/sep_correlation/` — SEP-to-semantic-entropy correlation data
- `results/assumption_verification/` — Calibration assumption verification data

---

## 🧪 Experimental Configuration

### Hardware

- **Edge Server**: NVIDIA RTX 5090, 64 GB RAM
- **Cloud API**: GPT-4 (OpenAI API)

### Datasets

| Dataset | Version | Task |
|---------|---------|------|
| MMLU | 2023-11 | 57-subject knowledge benchmark |
| TriviaQA | v1.0 | Open-domain QA |

### Models

| Role | Model | Details |
|------|-------|---------|
| Edge LLM | Phi-3-medium (14B) | Full precision |
| Edge LLM | Mistral-7B | Full precision |
| Cloud LLM | GPT-4 | OpenAI API |
| SEP Probe | 2-layer MLP | 128 hidden dims, <1 ms inference |

### Reproducibility

- **Random seeds**: {42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 6789}
- **Datasets**: Fixed HuggingFace versions

---

## 📦 Code Availability

> **The complete codebase (RACE scheduler, SEP training, baseline implementations) and SEP checkpoints will be released upon paper acceptance.**

Currently, this repository provides:
- ✅ Full experimental result data (`results/`)
- ✅ Result visualizations (`figures/`)
- ⏳ Source code — *available upon acceptance*
- ⏳ Pre-trained SEP checkpoints — *available upon acceptance*

---

## 📝 License

This project is licensed under the MIT License.

---

## 📧 Contact

For questions, please contact: **Zhihao Wang** (wangzhihao@sdufe.edu.cn)

**Acknowledgments**: This work was supported by the Humanities and Social Sciences Research Project of Ministry of Education of China (No. 23YJC630101).
