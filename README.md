# RiskLoop — Multi-Task Contract Risk Analysis Machine Learning Pipeline

**RiskLoop** is an engineering framework for evaluating multi-task vs. single-task representations in contract-risk analysis using the Contract Understanding Atticus Dataset (CUAD).

This project strictly adheres to the **RiskLoop Software Requirements Document (SRD), Version 1.1**.

---

## Core Research Question

> *Does a shared representation trained using a joint multi-task model (Condition B) provide a material practical benefit over separate single-task models (Condition A), without causing unacceptable regression on other tasks?*

---

## Controlled Experimental Protocol (12 Runs)

To isolate the representation variable without confounds, RiskLoop executes exactly **12 training runs**:

* **Condition A (Separate Single-Task Models):** 3 independent models $\times$ 3 random seeds = **9 runs**.
* **Condition B (Shared Multi-Task Model):** 1 shared model (backbone + 3 task heads) $\times$ 3 random seeds = **3 runs**.

### Validation-Only Adoption Decision Rules
1. **Stage 1 (Descriptive Reporting):** Report individual runs, mean, median, min, max, and range for all 3 validation seeds per task/condition.
2. **Stage 2A (Practical Improvement Rule):** Adopt Condition B for a task if and only if:
   $$\text{Median}(B) - \text{Median}(A) > 2.0 \text{ percentage points}$$
   **AND** at least 2 of Condition B's 3 validation runs individually exceed $\text{Max}(A)$. Otherwise, default to Condition A.
3. **Stage 2B (Cross-Task Non-Regression Veto):** Flag potential regression if $\text{Median}(A) - \text{Median}(B) > 0.5$ percentage points. Veto Condition B globally as a joint model **only** if the 0.5 point regression is met **AND** at least 2 of B's 3 validation runs fall strictly below $\text{Min}(A)$.
4. **Stage 2B Ambiguity:** Regressions $>0.5\%$ lacking extreme consistency are disclosed as ambiguous/inconclusive and do **NOT** auto-veto Condition B.

---

## Directory Structure

```
RiskLoop/
├── .gitignore
├── pyproject.toml
├── README.md
├── RiskLoop_SRD_v1.1.docx
├── config/
│   └── config_template.json
├── docs/
│   ├── software_design_document.md
│   └── traceability_matrix.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── experiments/
├── reports/
├── scripts/
│   ├── check_leakage.py
│   ├── run_data_audit.py
│   └── verify_config_drift.py
├── src/
│   └── riskloop/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── schema.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── splitter.py
│       │   ├── audit.py
│       │   └── leakage.py
│       ├── experiment/
│       │   ├── __init__.py
│       │   └── runner.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── adoption.py
│       │   ├── selection.py
│       │   └── test_evaluator.py
│       └── utils/
│           ├── __init__.py
│           └── reproducibility.py
└── tests/
    ├── conftest.py
    ├── test_split_leakage.py
    ├── test_split_deterministic.py
    ├── test_config_immutability.py
    ├── test_config_drift.py
    ├── test_experiment_integrity.py
    ├── test_data_audit.py
    ├── test_adoption_rules.py
    ├── test_model_selection.py
    └── test_test_set_isolation.py
```

---

## Engineering Constraints & Hard Rules

1. **Contract-Level Data Splitting:** Partitioning is performed strictly at the contract level. No derived chunk or span may cross partition boundaries.
2. **Pre-Training Sparsity Audit:** Enforces minimum thresholds ($\ge 25$ Train, $\ge 8$ Val, $\ge 8$ Test usable positive contracts) prior to training.
3. **Pre-Experiment Decision Lock:** All preprocessing, backbone, and evaluation parameters are locked into a SHA-256 hashed configuration prior to training.
4. **Test Set Isolation:** The test set is evaluated exactly once after all adoption decisions and representative model selections are frozen.

---

## Setup & Verification

1. Install package in editable mode with development dependencies:
   ```bash
   pip install -e .[dev]
   ```

2. Run the test suite:
   ```bash
   pytest -v
   ```

---

## Author & Specifications
* **Author:** Aayushi Gupta
* **Specification Baseline:** SRD v1.1 (July 27, 2026)
