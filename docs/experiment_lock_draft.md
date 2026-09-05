# RiskLoop — Pre-Experiment Lock Document

**Date:** September 5, 2026  
**Status:** **LOCKED FOR SDD & EXPERIMENT EXECUTION**  
**Requirements Baseline:** RiskLoop SRD v1.1 (FR-17 through FR-20)

---

## 1. Document Overview & Decision Status

This document records the **Pre-Experiment Decision Lock** required by RiskLoop SRD Section 5.1 (FR-17 to FR-20).

> [!IMPORTANT]
> **DECISION STATUS: LOCKED**  
> Following user approval and post-split verification (Train $\ge 25$, Val $\ge 8$, Test $\ge 8$ positive contracts), the selected task set, contract split map, and experimental protocol are officially **LOCKED**.

---

## 2. Selected Task Set & Risk Rationale (LOCKED)

The RiskLoop 12-run experiment will evaluate the following three selected tasks, providing coverage across distinct contractual risk dimensions:

1. **`Cap On Liability` (Liability / Exposure Risk):**
   - *Train Pos Contracts:* 198 ($\ge 25$) | *Val Pos Contracts:* 39 ($\ge 8$) | *Test Pos Contracts:* 38 ($\ge 8$)
   - *Rationale:* Captures financial risk allocation, liability caps, and claim timeframe limits (275 total positive contracts / 53.9% prevalence).
2. **`Anti-Assignment` (Assignment / Transfer Restriction Risk):**
   - *Train Pos Contracts:* 257 ($\ge 25$) | *Val Pos Contracts:* 61 ($\ge 8$) | *Test Pos Contracts:* 56 ($\ge 8$)
   - *Rationale:* Captures structural contract transfer risk and consent/notice requirements upon assignment (374 total positive contracts / 73.3% prevalence).
3. **`Termination For Convenience` (Termination / Exit-Right Risk):**
   - *Train Pos Contracts:* 123 ($\ge 25$) | *Val Pos Contracts:* 28 ($\ge 8$) | *Test Pos Contracts:* 32 ($\ge 8$)
   - *Rationale:* Captures commercial relationship exit risk and unilateral termination notice timelines (183 total positive contracts / 35.9% prevalence).

---

## 3. Contract Partitioning & Split Map (LOCKED)

- **Split Map Artifact:** [`data/splits/contract_split_map.json`](file:///c:/Users/aayushi/OneDrive/Documents/RiskLoop/data/splits/contract_split_map.json)
- **Split Parameters:** Seed `42`, 70% Train (357 contracts) / 15% Val (76 contracts) / 15% Test (77 contracts).
- **Leakage Audit Status:** **PASSED** (0 contract overlap across partitions).
- **Partitioning Unit:** Contracts are the sole independent partitioning unit; all annotations for contract $C_i$ remain strictly within $C_i$'s split.

---

## 4. Locked Protocol Rules (MUST NOT BE ALTERED)

1. **Controlled 12-Run Suite (FR-21..FR-24):**
   - Condition A: 3 independent single-task models $\times 3$ random seeds = **9 runs**.
   - Condition B: 1 shared multi-task model (backbone + 3 task heads) $\times 3$ random seeds = **3 runs**.
   - Total runs = **12**. No backbone, hyperparameter, or loss-weight sweeps permitted.
2. **Zero Contract-Level Data Leakage (FR-05..FR-07, NFR-12):**
   - Partitioning is strictly contract-level (`train`, `val`, `test`).
3. **Configuration Immutability & Hash Locking (FR-17..FR-20):**
   - Once frozen in Phase 4, configuration parameters generate a SHA-256 digest (`config_hash`).
   - Mid-experiment modification invalidates the experiment and requires a restart under a new versioned config.
4. **Validation-Only Architecture Adoption Protocol (FR-29..FR-34):**
   - **Stage 2A:** Adopt Condition B for task $t$ iff $\text{Median}(B_t) - \text{Median}(A_t) > 0.02$ AND $\ge 2$ B runs exceed $\max(A_t)$.
   - **Stage 2B Veto:** Veto Condition B globally iff $\text{Median}(A_t) - \text{Median}(B_t) > 0.005$ AND $\ge 2$ B runs fall strictly below $\min(A_t)$.
   - **Stage 2B Ambiguity:** Regressions $>0.5\%$ without extreme consistency are flagged as ambiguous, disclosed, and do **not** auto-veto Condition B.
5. **Representative Model Selection (FR-35..FR-37):**
   - Occurs strictly post-freeze. Condition A picks median validation run; Condition B picks seed with lowest sum of ranks across tasks.
6. **Single-Execution Test Set Isolation (FR-38..FR-40):**
   - Test set evaluated exactly once post-freeze. Test results cannot reopen decisions.

---

## 5. Remaining TBD Items (To Be Resolved in Phase 4 Configuration Lock)

1. Tokenizer choice and maximum sequence length.
2. Chunking stride and span-alignment logic.
3. Positive example construction and negative sampling ratio.
4. Task-specific decision thresholds and primary evaluation metrics.
