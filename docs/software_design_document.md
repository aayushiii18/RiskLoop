# RiskLoop — Software Design Document (SDD)
**Version:** 1.0  
**Status:** DRAFT — BASELINE FOR PHASE 1 FOUNDATION  
**Author:** AI Coding Agent (Antigravity) / Aayushi Gupta  
**Date:** August 26, 2026  
**Requirements Baseline:** RiskLoop SRD v1.1 (July 27, 2026)

---

## 1. System Overview

RiskLoop is a multi-task contract-risk analysis machine learning system built on the Contract Understanding Atticus Dataset (CUAD). The primary objective of the system is to answer a fundamental architectural research question:

> *Does a shared representation trained using a joint multi-task model (Condition B) provide a material practical benefit over separate single-task models (Condition A), without causing unacceptable regression on other tasks?*

The system enforces strict scientific rigor, zero contract-level data leakage, deterministic reproducibility, pre-experiment configuration locking, validation-only policy-based architecture adoption, and single-execution test-set isolation.

---

## 2. High-Level Architecture

The RiskLoop system consists of five primary subsystem layers:

1. **Data Management Subsystem:**
   - **Parser / Task Discovery Engine:** Interfaces with `CUAD_v1.json` (Phase 2).
   - **Contract-Level Splitter:** Partitions contracts into Train, Validation, and Test sets.
   - **Data-Sparsity & Adequacy Auditor:** Evaluates per-task/per-split contract counts against adequacy thresholds ($\ge 25$ Train, $\ge 8$ Val, $\ge 8$ Test usable positive contracts).
   - **Leakage Verifier:** Programmatically inspects split boundaries for zero contract overlap.

2. **Configuration Management Subsystem:**
   - **Schema & Immutability Engine:** Defines locked configuration fields and calculates SHA-256 digests.
   - **Drift Verifier:** Audits Condition A and Condition B configurations to ensure representation isolation without parameter drift.

3. **Experiment Orchestration Subsystem:**
   - **12-Run Orchestrator:** Manages execution of 9 Condition A runs (3 single-task models $\times 3$ seeds) and 3 Condition B runs (1 joint model $\times 3$ seeds).

4. **Evaluation & Adoption Subsystem:**
   - **Metrics Engine:** Calculates precision, recall, F1, macro-F1, confusion matrix, ROC-AUC, and PR-AUC.
   - **Validation Adoption Protocol Engine:** Executes Stage 1 reporting, Stage 2A improvement math (+2.0 point margin + extreme consistency), Stage 2B regression veto (+0.5 point margin + extreme consistency), and Stage 2B ambiguity handling.
   - **Representative Model Selector:** Deterministically selects median run for Condition A and lowest sum-of-ranks seed for Condition B.

5. **Test Evaluation Subsystem:**
   - **Post-Freeze Test Evaluator:** Evaluates selected representative models on the untouched test set exactly once.

---

## 3. Data Flow

```
+------------------+
|   CUAD_v1.json   |
+--------+---------+
         |
         v
+------------------+     +--------------------------+
| Contract Splitter| --> | contract_split_map.json  |
+--------+---------+     +------------+-------------+
         |                            |
         v                            v
+---------------------------------------------------+
|             Data Sparsity Audit                   |
| (Gates: Train >= 25, Val >= 8, Test >= 8 pos cnts)|
+--------+------------------------------------------+
         | (Pass)
         v
+---------------------------------------------------+
|         Pre-Experiment Decision Lock              |
|        (locked_config.json + SHA256 Hash)          |
+--------+------------------------------------------+
         |
         +----------------------------+
         |                            |
         v                            v
+------------------+        +-------------------+
|   Condition A    |        |    Condition B    |
| 3 tasks x 3 seeds|        | 1 joint x 3 seeds |
|    (9 runs)      |        |     (3 runs)      |
+--------+---------+        +---------+---------+
         |                            |
         +--------------+-------------+
                        |
                        v
         +-----------------------------+
         |  Validation Adoption Engine |
         |   (Stage 1, 2A, 2B Math)    |
         +--------------+--------------+
                        | (Decision Frozen)
                        v
         +-----------------------------+
         | Representative Model Select |
         |  (Cond A Median / B Ranks)  |
         +--------------+--------------+
                        | (Models Frozen)
                        v
         +-----------------------------+
         |    Test Set Evaluator       |
         |   (Executed EXACTLY ONCE)   |
         +-----------------------------+
```

---

## 4. Module Responsibilities

| Module Path | Primary Responsibility | SRD Requirement Mapping |
| :--- | :--- | :--- |
| `src/riskloop/data/splitter.py` | Partitions contracts deterministically into Train/Val/Test. | FR-05, FR-06, FR-07 |
| `src/riskloop/data/leakage.py` | Detects any contract overlap across partitions. | FR-06, NFR-12 |
| `src/riskloop/data/audit.py` | Runs sparsity audit, checks adequacy thresholds, outputs diagnostics. | FR-08..FR-13, NFR-10 |
| `src/riskloop/config/schema.py` | Manages locked configuration, SHA-256 hashing, and drift verification. | FR-17..FR-20, NFR-08, NFR-13 |
| `src/riskloop/experiment/runner.py` | Orchestrates the 12 experiment runs (9 single-task, 3 multi-task). | FR-21..FR-24 |
| `src/riskloop/evaluation/metrics.py` | Computes standard per-task and multi-task evaluation metrics. | FR-25 |
| `src/riskloop/evaluation/adoption.py` | Implements Stage 1, Stage 2A (+2% margin), Stage 2B (0.5% veto/ambiguity). | FR-29..FR-34 |
| `src/riskloop/evaluation/selection.py` | Implements Condition A median selection and Condition B sum-of-ranks. | FR-35..FR-37 |
| `src/riskloop/evaluation/test_evaluator.py` | Executes test evaluation exactly once on frozen representative models. | FR-38..FR-40 |
| `src/riskloop/utils/reproducibility.py` | Injects fixed random seeds across Python, NumPy, and PyTorch. | NFR-06, NFR-07 |

---

## 5. Experiment Orchestration Concept

The experiment orchestrator (`runner.py`) enforces strict scientific control over the 12-run suite:

1. **Pre-flight Checks:**
   - Validates that `locked_config.json` exists and its SHA-256 hash matches the frozen digest.
   - Verifies that data-sparsity audit passed for all candidate tasks.
   - Audits Condition A and Condition B parameters to ensure zero configuration drift (NFR-13).

2. **Execution Matrix (12 Runs):**
   - **Condition A (9 Runs):**
     - Task 1: Seeds $S_1, S_2, S_3$ (3 runs)
     - Task 2: Seeds $S_1, S_2, S_3$ (3 runs)
     - Task 3: Seeds $S_1, S_2, S_3$ (3 runs)
   - **Condition B (3 Runs):**
     - Joint Model (Tasks 1, 2, 3): Seeds $S_1, S_2, S_3$ (3 runs)

3. **Output Isolation:** Each run writes its outputs to an isolated directory structure tagged by condition, task, and seed identifier.

---

## 6. Configuration Management

Configuration is defined via Pydantic models in `src/riskloop/config/schema.py`.

### Locked Configuration Structure
- **Experiment Hash:** SHA-256 calculation over serialized JSON config.
- **Immutability Policy:** Any modification to hyperparameter fields invalidates the active run and requires generating a new versioned configuration file (FR-20).
- **Drift Protection:** Condition A and Condition B configs must match on backbone, learning rate, batch size, max sequence length, chunking stride, tokenizer, and random seeds.

---

## 7. Evaluation Architecture & Adoption Logic

The adoption engine (`adoption.py`) executes a two-stage evaluation protocol using validation data **only**:

### Stage 1: Descriptive Reporting (FR-30)
Computes individual values, mean, median, min, max, and range for validation metric scores across the 3 seeds for each task under Condition A and Condition B.

### Stage 2A: Practical Improvement Decision Rule (FR-31)
For task $t$, Condition B is adopted if and only if:
$$\text{Median}(B_t) - \text{Median}(A_t) > 0.02$$
$$\text{AND } \sum_{i=1}^{3} \mathbb{I}(B_{t,i} > \max(A_t)) \ge 2$$
If this dual condition is met, Condition B is adopted for task $t$. Otherwise, task $t$ defaults to Condition A.

### Stage 2B: Cross-Task Non-Regression Veto (FR-32..FR-34)
A potential regression is flagged for task $t$ if:
$$\text{Median}(A_t) - \text{Median}(B_t) > 0.005$$

- **Global Veto (FR-33):** Condition B is vetoed as a joint model **only** if the 0.5-point regression is flagged **AND** extreme consistency holds:
  $$\sum_{i=1}^{3} \mathbb{I}(B_{t,i} < \min(A_t)) \ge 2$$
  If vetoed, all tasks default to Condition A regardless of Stage 2A results.
- **Ambiguity (FR-34):** If $\text{Median}(A_t) - \text{Median}(B_t) > 0.005$ but fewer than 2 B runs fall below $\min(A_t)$, the regression is classified as **inconclusive / ambiguous**. It is logged and disclosed, but does **not** auto-veto Condition B.

---

## 8. Reproducibility Strategy

1. **Global Seed Locking:** `set_seed(seed)` sets random seeds for `random`, `numpy`, `torch`, and `torch.cuda` (NFR-06, NFR-07).
2. **Deterministic Split Mapping:** `splitter.py` uses a fixed seed to partition contracts and outputs `contract_split_map.json` (FR-07).
3. **Configuration Hashes:** SHA-256 checksum recorded in experiment metadata (NFR-08).

---

## 9. Leakage-Prevention Strategy

1. **Contract-Level Isolation:** `contract_split_map.json` assigns contract filenames/IDs to `train`, `val`, or `test`.
2. **Child Example Inheritance:** All spans, paragraphs, or chunks derived from contract $C_i$ inherit $C_i$'s assigned split partition.
3. **Pre-Training Verification:** `leakage.py` validates that:
   $$\text{Set}(\text{Contracts}_{\text{train}}) \cap \text{Set}(\text{Contracts}_{\text{val}}) = \emptyset$$
   $$\text{Set}(\text{Contracts}_{\text{train}}) \cap \text{Set}(\text{Contracts}_{\text{test}}) = \emptyset$$
   $$\text{Set}(\text{Contracts}_{\text{val}}) \cap \text{Set}(\text{Contracts}_{\text{test}}) = \emptyset$$
   Any violation causes an immediate hard execution halt (NFR-12).

---

## 10. Test-Set Isolation Strategy

1. **Single Execution Enforcement:** The test set is evaluated strictly once via `test_evaluator.py` (FR-39).
2. **Freeze Dependency Gate:** `test_evaluator.py` requires a signed `adoption_decision.json` artifact indicating that architecture adoption and representative model selection are finalized.
3. **No Retroactive Decision Modification:** Results from test evaluation cannot trigger model re-selection or alter the frozen architecture decision (FR-40).

---

## 11. TBD Register (Data-Dependent Decisions)

Per SRD Section 8, the following items remain strictly **TBD** pending actual inspection of `CUAD_v1.json` during Phase 2:

1. **Exact Task Names & Question Semantics:** Pending inspection of CUAD question categories and positive sample distributions.
2. **Tokenizer & Sequence Length:** Pending text length distributions of CUAD contract clauses.
3. **Chunking Strategy & Overlap Stride:** Pending span length analysis.
4. **Positive/Negative Example Construction:** Pending span boundary overlap rules and negative sample ratio audit.
5. **Primary Metric Selection:** Pending class imbalance and prevalence audit across candidate tasks.
6. **Task Decision Thresholds:** Pending validation precision-recall trade-off analysis during Pre-Experiment Lock.
