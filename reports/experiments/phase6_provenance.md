# RiskLoop Phase 6 Experiment Provenance & Artifact Record

**Commit HEAD:** `19cfaf21dddc0013296a56366e2bbcdb3cc60b4e`  
**Protocol Reference:** RiskLoop SRD v1.1 Phase 5C Decision Protocol (FR-29..FR-37, NFR-01..NFR-04)  
**Test Set Path:** `data/processed/test_chunks.json`  
**Test Set SHA256:** `44c42d672685d1851efe33e5aecb6f603e1c2e1446b99bc33e985fc64d56d185`  

---

## 1. Original Checkpoint Availability Status

> [!IMPORTANT]  
> The original representative Phase 5B checkpoint binaries are unavailable. Therefore, a byte-for-byte or weight-for-weight reproduction of the historical Phase 6 test inference is not possible.

* **Availability Status:** `UNAVAILABLE`
* **Storage Notes:** Original Phase 5B representative model binaries (`run_03`, `run_04`, `run_07`) were output to ephemeral container storage (`/kaggle/working/RiskLoop/reports/experiments/run_XX/`) and were not persisted to Git or an external artifact repository.

---

## 2. Protocol-Faithful Reproduction Status

> [!NOTE]  
> The reproduced checkpoints are evaluated separately as a protocol-faithful reproduction experiment and must not be interpreted as the historical Phase 6 test results.

* **Reproduction Status:** `PROTOCOL_FAITHFUL_REPRODUCTION`
* **Execution Strategy:** Reproduced models are generated from scratch using identical locked training code, seeds, data splits, and hyperparameters, and are saved to isolated output directories.

---

## 3. Historical Phase 5C Validation & Model Selection Results (Preserved)

* **Decision Artifact:** [`reports/experiments/decision_analysis.json`](file:///c:/Users/aayushi/OneDrive/Documents/RiskLoop/reports/experiments/decision_analysis.json)
* **Joint Architecture Vetoed:** `True` (Condition B triggered Stage 2B global veto across all 3 tasks)
* **Final Task Adoptions:** All tasks adopted `Condition_A` single-task models.

### Historical Representative Models
1. **Cap On Liability:** `Condition_A` \| `run_03` \| Seed `44` \| Historical Val F1: `0.791837` (Best Epoch 3)
2. **Anti-Assignment:** `Condition_A` \| `run_04` \| Seed `42` \| Historical Val F1: `0.771831` (Best Epoch 2)
3. **Termination For Convenience:** `Condition_A` \| `run_07` \| Seed `42` \| Historical Val F1: `0.666667` (Best Epoch 2)

---

## 4. Test Set Isolation Guarantee

TEST SET ISOLATION STRICTLY ENFORCED: The test set (data/processed/test_chunks.json) has NOT been opened, loaded, evaluated, or accessed in any way during this decision analysis (FR-38).
