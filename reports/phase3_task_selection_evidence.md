# RiskLoop Phase 3A.2 — Evidence-Based Task Selection Analysis Report
**Date:** September 5, 2026  
**Dataset Base:** CUAD v1 (`aok_v1.0`, 510 contracts)  
**Requirements Baseline:** RiskLoop SRD v1.1 (FR-01 through FR-13, FR-18)  
**Status:** RECOMMENDATION SUBMITTED FOR REVIEW (Pre-Lock)

---

## 1. Executive Summary
This report presents the Phase 3A.2 evidence-based evaluation of candidate CUAD tasks for the **RiskLoop** multi-task contract-risk machine learning system.

### Key Findings & Recommendation
1. **Recommendation:** **KEEP AND MODIFY THE FORMULATION OF THE CURRENT THREE RISK TASKS**.
   - **Task 1: Liability Risk** (`Cap On Liability` + `Uncapped Liability`) — Financial Exposure Risk Domain
   - **Task 2: Insurance Risk** (`Insurance`) — Operational Compliance Risk Domain
   - **Task 3: Change Of Control Risk** (`Change Of Control`) — Corporate Transaction / M&A Risk Domain
2. **Formulation Modification Required:** Analysis of `data/raw/CUAD_v1.json` reveals that **100% of contracts containing `Uncapped Liability` (111/111) ALSO contain `Cap On Liability`**. Formulating a single mutually-exclusive 3-class target (`capped | uncapped | neither`) at the contract level creates a label contradiction for 111 contracts (21.8% of the dataset). The Liability task head should instead be formulated as multi-label binary classification (`has_liability_cap`, `has_uncapped_liability`) or evaluating `Cap On Liability` as primary liability risk while tracking `Uncapped Liability` as a sub-clause attribute.
3. **Risk Domain Diversity:** Selecting Liability, Insurance, and Change Of Control provides high domain diversity across financial risk, operational risk, and corporate transaction risk, whereas alternative high-prevalence tasks (`License Grant`, `Anti-Assignment`) overlap heavily with `Cap On Liability` (Jaccard similarity > 0.50).

---

## 2. Empirical Co-Occurrence Analysis: Liability Task

| Metric / Category | Contract Count | Percentage of CUAD (N=510) |
| :--- | :---: | :---: |
| Contracts with `Cap On Liability` | 275 | 53.9% |
| Contracts with `Uncapped Liability` | 111 | 21.8% |
| Contracts with **BOTH** `Cap` & `Uncapped` | 111 | 21.8% |
| Contracts with `Cap` ONLY | 164 | 32.2% |
| Contracts with `Uncapped` ONLY | 0 | 0.0% |
| Contracts with NEITHER | 235 | 46.1% |

> [!IMPORTANT]
> **Fact vs Inference:** Zero contracts in CUAD contain `Uncapped Liability` without also containing `Cap On Liability`. Therefore, `Uncapped Liability` is a strict sub-clause phenomenon of contracts that contain liability caps. Treating `capped` and `uncapped` as mutually exclusive contract classes is methodologically invalid.

---

## 3. Comprehensive Candidate Tasks Comparison Table

| Candidate Task Name | Pos Contracts | Prevalence | Pos Spans | Mean Spans/Pos Cntr | Med Spans | Max Spans | Top 10% Share | Top 20% Share | Pre-Split Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Cap On Liability` | 275/510 | 53.9% | 672 | 2.4 | 2 | 16 | 29.0% | 45.4% | PASS |
| `Uncapped Liability` | 111/510 | 21.8% | 167 | 1.5 | 1 | 5 | 25.7% | 39.5% | PASS |
| `Insurance` | 166/510 | 32.5% | 560 | 3.4 | 2 | 17 | 35.9% | 52.7% | PASS |
| `Change Of Control` | 121/510 | 23.7% | 253 | 2.1 | 2 | 11 | 27.3% | 42.3% | PASS |
| `License Grant` | 255/510 | 50.0% | 777 | 3.0 | 2 | 16 | 30.9% | 49.7% | PASS |
| `Anti-Assignment` | 374/510 | 73.3% | 654 | 1.7 | 1 | 10 | 25.1% | 41.6% | PASS |
| `Exclusivity` | 180/510 | 35.3% | 410 | 2.3 | 2 | 12 | 27.1% | 42.0% | PASS |
| `Non-Compete` | 119/510 | 23.3% | 259 | 2.2 | 2 | 12 | 32.0% | 46.7% | PASS |
| `Post-Termination Services` | 182/510 | 35.7% | 450 | 2.5 | 2 | 23 | 36.7% | 52.0% | PASS |
| `Audit Rights` | 214/510 | 42.0% | 643 | 3.0 | 2 | 19 | 33.6% | 50.5% | PASS |
| `Termination For Convenience` | 183/510 | 35.9% | 246 | 1.3 | 1 | 4 | 20.7% | 35.4% | PASS |
| `Ip Ownership Assignment` | 124/510 | 24.3% | 318 | 2.6 | 2 | 15 | 27.0% | 43.1% | PASS |
| `Renewal Term` | 176/510 | 34.5% | 210 | 1.2 | 1 | 5 | 21.4% | 33.3% | PASS |

---

## 4. Ranked Shortlist & Risk Domain Analysis

1. **Rank 1 — Liability Risk** (`Cap On Liability` & `Uncapped Liability`) — *Financial Exposure Risk*
   - **Data Sufficiency:** 275 positive contracts (53.9%) for Cap On Liability; 111 positive contracts (21.8%) for Uncapped Liability.
   - **Justification:** Fundamental risk domain for any contract analysis system. Easily passes pre-split feasibility screen.
2. **Rank 2 — Insurance Risk** (`Insurance`) — *Operational Compliance Risk*
   - **Data Sufficiency:** 166 positive contracts (32.5%), 560 spans.
   - **Justification:** Distinct operational compliance domain; low pairwise overlap with corporate tasks (Jaccard 0.25).
3. **Rank 3 — Change Of Control Risk** (`Change Of Control`) — *Corporate Governance / M&A Risk*
   - **Data Sufficiency:** 121 positive contracts (23.7%), 253 spans.
   - **Justification:** Corporate transaction risk domain; lowest pairwise overlap across all candidates (Jaccard 0.20 to 0.30); clean span density (mean 2.1, max 11).
4. **Rank 4 (Alternative) — Termination For Convenience** (`Termination For Convenience`) — *Commercial Duration Risk*
   - **Data Sufficiency:** 183 positive contracts (35.9%), 246 spans.
   - **Justification:** Outstanding low concentration (top 10% share 20.7%, max 4 spans).
5. **Rank 5 (Alternative) — Audit Rights** (`Audit Rights`) — *Operational Oversight Risk*
   - **Data Sufficiency:** 214 positive contracts (42.0%), 643 spans.
6. **Rank 6 (Alternative) — License Grant** (`License Grant`) — *IP / Licensing Scope Risk*
   - **Data Sufficiency:** 255 positive contracts (50.0%), 777 spans.

---

## 5. Missing Evidence Register & Post-Split Requirements

> [!WARNING]
> **Post-Split Verification Required:** The pre-split count ($\ge 41$ positive contracts) is ONLY a feasibility screen. The task set cannot be officially locked until the following evidence is verified:
1. **Post-Split Contract Partition Check:** After generating `contract_split_map.json`, verify that Train $\ge 25$, Val $\ge 8$, Test $\ge 8$ positive contracts are achieved for each of the 3 selected tasks.
2. **Sequence Length & Chunk Stride:** Character/token length distributions of contract clauses for the selected tasks.
3. **Thresholds & Primary Metrics:** Validation PR curve analysis to lock decision thresholds and primary metrics.