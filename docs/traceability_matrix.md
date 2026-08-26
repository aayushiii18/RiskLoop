# RiskLoop — Requirements Traceability Matrix

This document maps all Functional Requirements (FR-01 to FR-40) and Non-Functional Requirements (NFR-01 to NFR-13) from **RiskLoop SRD v1.1** to their corresponding Software Design Document (SDD) modules and automated test suite verifications.

---

## Functional Requirements (FR)

| Req ID | Requirement Summary | Implementation Module | Verification Method / Test Case | Status |
| :--- | :--- | :--- | :--- | :--- |
| **FR-01** | Use `CUAD_v1.json` as sole source of truth | `src/riskloop/data/discovery.py` | Manual audit of parser logs | Pending Phase 2 |
| **FR-02** | No task invention before CUAD inspection | SDD Section 11 (TBD Register) | Inspection of project configs | Baseline Established |
| **FR-03** | Task definitions derived from CUAD annotations | `src/riskloop/data/discovery.py` | Data audit report inspection | Pending Phase 2 |
| **FR-04** | Task viability determined by sparsity audit | `src/riskloop/data/audit.py` | `tests/test_data_audit.py` | Module Implemented |
| **FR-05** | Contract assigned to exactly 1 split | `src/riskloop/data/splitter.py` | `tests/test_split_leakage.py` | Module Implemented |
| **FR-06** | Zero contract-level split leakage | `src/riskloop/data/leakage.py` | `tests/test_split_leakage.py` | Module Implemented |
| **FR-07** | Deterministic split map saved as JSON | `src/riskloop/data/splitter.py` | `tests/test_split_deterministic.py` | Module Implemented |
| **FR-08** | Execute sparsity audit before training | `src/riskloop/data/audit.py` | `tests/test_data_audit.py` | Module Implemented |
| **FR-09** | Report audit per-task/per-split metrics | `src/riskloop/data/audit.py` | `tests/test_data_audit.py` | Module Implemented |
| **FR-10** | Report contract concentration diagnostics | `src/riskloop/data/audit.py` | `tests/test_data_audit.py` | Module Implemented |
| **FR-11** | Minimum contract thresholds (25/8/8) | `src/riskloop/data/audit.py` | `tests/test_data_audit.py` | Module Implemented |
| **FR-12** | Task filtering and audit decision logic | `src/riskloop/data/audit.py` | `tests/test_data_audit.py` | Module Implemented |
| **FR-13** | Audit distinguishes failure causes | `src/riskloop/data/audit.py` | `tests/test_data_audit.py` | Module Implemented |
| **FR-14** | Preprocessing governed by config lock | `src/riskloop/config/schema.py` | `tests/test_config_immutability.py` | Module Implemented |
| **FR-15** | Raw model output named `model_score` | `src/riskloop/evaluation/metrics.py` | Code inspection & schema test | Module Implemented |
| **FR-16** | No probability calibration for `model_score` | `src/riskloop/evaluation/metrics.py` | Metric engine inspection | Baseline Established |
| **FR-17** | Finalize & freeze config pre-training | `src/riskloop/config/schema.py` | `tests/test_config_immutability.py` | Module Implemented |
| **FR-18** | Required locked variables list | `src/riskloop/config/schema.py` | `tests/test_config_immutability.py` | Module Implemented |
| **FR-19** | Config machine-readable & immutable | `src/riskloop/config/schema.py` | `tests/test_config_immutability.py` | Module Implemented |
| **FR-20** | Mid-experiment bug invalidates experiment | `src/riskloop/config/schema.py` | Hash check in runner | Module Implemented |
| **FR-21** | Isolate variable, execute 12 runs | `src/riskloop/experiment/runner.py` | `tests/test_experiment_integrity.py` | Module Implemented |
| **FR-22** | Condition A = 3 tasks $\times$ 3 seeds (9 runs) | `src/riskloop/experiment/runner.py` | `tests/test_experiment_integrity.py` | Module Implemented |
| **FR-23** | Condition B = 1 joint $\times$ 3 seeds (3 runs) | `src/riskloop/experiment/runner.py` | `tests/test_experiment_integrity.py` | Module Implemented |
| **FR-24** | No additional parameter sweeps | `src/riskloop/experiment/runner.py` | `tests/test_config_drift.py` | Module Implemented |
| **FR-25** | Evaluation metrics set reported | `src/riskloop/evaluation/metrics.py` | `tests/test_adoption_rules.py` | Module Implemented |
| **FR-26** | Primary metric selected post-audit | SDD Section 11 (TBD Register) | Pre-experiment config audit | Pending Phase 2 |
| **FR-27** | Metric selection balances imbalance/costs | `src/riskloop/evaluation/metrics.py` | Design review | Pending Phase 2 |
| **FR-28** | Primary metric identical for A and B | `src/riskloop/config/schema.py` | `tests/test_config_drift.py` | Module Implemented |
| **FR-29** | Validation-only adoption, no p-values | `src/riskloop/evaluation/adoption.py` | `tests/test_adoption_rules.py` | Module Implemented |
| **FR-30** | Stage 1 descriptive reporting | `src/riskloop/evaluation/adoption.py` | `tests/test_adoption_rules.py` | Module Implemented |
| **FR-31** | Stage 2A practical improvement (+2%) | `src/riskloop/evaluation/adoption.py` | `tests/test_adoption_rules.py` | Module Implemented |
| **FR-32** | Stage 2B potential regression flag (+0.5%) | `src/riskloop/evaluation/adoption.py` | `tests/test_adoption_rules.py` | Module Implemented |
| **FR-33** | Stage 2B regression veto rule | `src/riskloop/evaluation/adoption.py` | `tests/test_adoption_rules.py` | Module Implemented |
| **FR-34** | Stage 2B ambiguity disclosure | `src/riskloop/evaluation/adoption.py` | `tests/test_adoption_rules.py` | Module Implemented |
| **FR-35** | Model selection occurs post-freeze | `src/riskloop/evaluation/selection.py` | `tests/test_model_selection.py` | Module Implemented |
| **FR-36** | Condition A median seed selection | `src/riskloop/evaluation/selection.py` | `tests/test_model_selection.py` | Module Implemented |
| **FR-37** | Condition B joint sum-of-ranks selection | `src/riskloop/evaluation/selection.py` | `tests/test_model_selection.py` | Module Implemented |
| **FR-38** | Test set untouched during design | `src/riskloop/evaluation/test_evaluator.py` | `tests/test_test_set_isolation.py` | Module Implemented |
| **FR-39** | Test set evaluated exactly once | `src/riskloop/evaluation/test_evaluator.py` | `tests/test_test_set_isolation.py` | Module Implemented |
| **FR-40** | Test results cannot reopen decisions | `src/riskloop/evaluation/test_evaluator.py` | `tests/test_test_set_isolation.py` | Module Implemented |

---

## Non-Functional Requirements (NFR)

| Req ID | Requirement Summary | Implementation Module | Verification Method / Test Case | Status |
| :--- | :--- | :--- | :--- | :--- |
| **NFR-01** | Descriptive evidence statement | `src/riskloop/evaluation/adoption.py` | Report generation review | Baseline Established |
| **NFR-02** | No confidence interval on stochasticity | `src/riskloop/evaluation/adoption.py` | Code inspection | Baseline Established |
| **NFR-03** | No hypothesis tests / p-values | `src/riskloop/evaluation/adoption.py` | `tests/test_adoption_rules.py` | Module Implemented |
| **NFR-04** | No seed pairing assumption | `src/riskloop/evaluation/adoption.py` | Code & adoption logic check | Module Implemented |
| **NFR-05** | Validation bootstrapping constraint | `src/riskloop/evaluation/metrics.py` | Metric engine inspection | Baseline Established |
| **NFR-06** | Hardware/pipeline determinism | `src/riskloop/utils/reproducibility.py` | `tests/test_split_deterministic.py` | Module Implemented |
| **NFR-07** | Fixed random seed logging & injection | `src/riskloop/utils/reproducibility.py` | `tests/test_split_deterministic.py` | Module Implemented |
| **NFR-08** | Full pipeline version & config recording | `src/riskloop/config/schema.py` | `tests/test_config_immutability.py` | Module Implemented |
| **NFR-09** | Exportable descriptive statistics | `src/riskloop/evaluation/adoption.py` | JSON report export check | Module Implemented |
| **NFR-10** | Hard-fail on global audit failure | `src/riskloop/data/audit.py` | `tests/test_data_audit.py` | Module Implemented |
| **NFR-11** | Safe zero-prediction/undefined metrics | `src/riskloop/evaluation/metrics.py` | `tests/test_adoption_rules.py` | Module Implemented |
| **NFR-12** | Programmatic split leakage detection | `src/riskloop/data/leakage.py` | `tests/test_split_leakage.py` | Module Implemented |
| **NFR-13** | Configuration drift detection | `src/riskloop/config/schema.py` | `tests/test_config_drift.py` | Module Implemented |
