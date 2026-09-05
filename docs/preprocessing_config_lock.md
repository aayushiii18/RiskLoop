# RiskLoop — Preprocessing Configuration Lock Proposal Document

**Date:** September 5, 2026  
**Status:** FROZEN AND IMPLEMENTED (Phase 4C Complete)  
**Requirements Baseline:** RiskLoop SRD v1.1 (FR-05..FR-07, FR-14..FR-16, FR-17..FR-20)  
**Locked Inputs:**
- Selected Task Set: `Cap On Liability`, `Anti-Assignment`, `Termination For Convenience`
- Canonical Split Map: [`data/splits/contract_split_map.json`](file:///c:/Users/aayushi/OneDrive/Documents/RiskLoop/data/splits/contract_split_map.json) (Seed 42, 357 Train / 76 Val / 77 Test)

---

## 1. Phase 4A Correction Record

- **Corrected Fact:** 509 out of 510 contracts (**99.8%**) exceed standard 512-token sequence limits. The shortest contract in the dataset (`RMRGROUPINC_01_22_2020-EX-99.1-JOINT FILING AGREEMENT`) is 645 characters (~160 tokens) and fits within a single window.
- **Windowing Necessity:** Because the median contract length is 33,143 characters (~10,000 tokens) and the 90th percentile is 122,117 characters (~38,000 tokens), sliding-window chunking remains **strictly mandatory** for complete dataset coverage.

---

## 2. Actual Legal-BERT Token Counts for Positive Spans (1,572 Spans)

Measured directly using the official `nlpaueb/legal-bert-base-uncased` FastTokenizer across all 1,572 positive spans:

| Task Name | Pos Spans | Min Tokens | Median Tokens | Mean Tokens | 95th Pctile Tokens | 99th Pctile Tokens | Max Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Cap On Liability`** | 672 | 7 | **61** | 73.5 | **171** | **258** | 379 |
| **`Anti-Assignment`** | 654 | 1 | **41** | 55.4 | **136** | **232** | 439 |
| **`Termination For Convenience`** | 246 | 6 | **33** | 44.7 | **114** | **192** | 397 |

### Window Containment Verification (512 Max Sequence Length / 256 Token Stride)
- **Total Positive Spans Evaluated:** 1,572
- **Fully Contained in $\ge 1$ Window Chunk:** **1,566 spans (99.62%)**
- **Not Fully Contained (Crosses Window Boundary):** **6 spans (0.38%)**
- **Verification:** The 99.62% span containment confirmed under the actual Legal-BERT FastTokenizer exceeds the 99.3% preliminary estimate.

---

## 3. Multi-Task Chunk Label Distribution (All 510 Contracts = 20,971 Total Chunks)

Under 512 max length and 256 stride tokenization across the 510 contracts:

| Task Name | Total Chunks | Positive Chunks | Negative Chunks | Positive % | Negative % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Cap On Liability`** | 20,971 | 813 | 20,158 | **3.9%** | **96.1%** |
| **`Anti-Assignment`** | 20,971 | 886 | 20,085 | **4.2%** | **95.8%** |
| **`Termination For Convenience`** | 20,971 | 381 | 20,590 | **1.8%** | **98.2%** |

### Union Distribution Across All 3 Tasks
- **Chunks Positive for $\ge 1$ Task (Union Positive):** **1,988 chunks (9.5%)**
- **Chunks Negative for ALL 3 Tasks (Union Negative):** **18,983 chunks (90.5%)**

---

## 4. Negative Sampling Sensitivity Analysis (Train Split = 357 Contracts, 14,602 Total Chunks)

Evaluating **Option A** (all training chunks) vs **Option B** (all union-positive chunks + deterministic sampling of $K$ negative chunks per contract):

| Sampling Option | Total Retained Train Chunks | Compute Reduction vs Option A | Union Pos Chunks (%) | Cap On Liability Pos (%) | Anti-Assignment Pos (%) | Termination For Convenience Pos (%) | Unused Candidate Neg Chunks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Option A (All Chunks)** | 14,602 | 0.0% | 1,387 (9.5%) | 591 (4.0%) | 605 (4.1%) | 254 (1.7%) | 0 |
| **Option B ($K=2$)** | 2,088 | 85.7% | 1,387 (66.4%) | 591 (28.3%) | 605 (29.0%) | 254 (12.2%) | 12,514 |
| **Option B ($K=3$)** | 2,422 | 83.4% | 1,387 (57.3%) | 591 (24.4%) | 605 (25.0%) | 254 (10.5%) | 12,180 |
| **Option B ($K=4$) [RECOMMENDED]** | **2,745** | **81.2%** | **1,387 (50.5%)** | **591 (21.5%)** | **605 (22.0%)** | **254 (9.3%)** | **11,857** |
| **Option B ($K=5$)** | 3,062 | 79.0% | 1,387 (45.3%) | 591 (19.3%) | 605 (19.8%) | 254 (8.3%) | 11,540 |

### Sampling Unit Rationale & Recommendation
- **Recommended Sampling Unit:** **Option B ($K=4$ negative chunks per contract)**.
- **Multi-Task Rationale:** Contracts are the fundamental independent unit of data partitioning. Sampling $K=4$ negative chunks per contract guarantees that every contract—including contracts with 0 positive spans for a given task—contributes negative context chunks for that task head.
- **Class Balance & Throughput:** Reduces training set size from 14,602 to 2,745 chunks ($\sim 81.2\%$ speedup per epoch), while preserving 100% of union-positive chunks (1,387 chunks) and maintaining balanced per-task positive label ratios ($\sim 21.5\%$ Cap On Liability, $\sim 22.0\%$ Anti-Assignment, $\sim 9.3\%$ Termination For Convenience).

---

## 5. Sampling Reproducibility Specification

1. **Deterministic Candidate Ordering:** Negative candidate chunks for contract $C_i$ are ordered strictly by `(contract_title, chunk_idx)`.
2. **Fixed Random Seed:** `seed = 42` injected into `random.Random(42)`.
3. **Static Preprocessed Artifact:** Sampling occurs once during dataset preprocessing; the resulting chunk set is saved to `data/processed/train_chunks.json` and reused statically across all training epochs (zero epoch-dependent drift).

---

## 6. Preprocessing Configuration Parameter Table

| Parameter Name | Proposed Value | Origin Label | Status |
| :--- | :--- | :--- | :---: |
| `selected_tasks` | `["Cap On Liability", "Anti-Assignment", "Termination For Convenience"]` | **SRD-mandated / Phase 3 Locked** | **APPROVED** |
| `contract_split_map` | `data/splits/contract_split_map.json` | **SRD-mandated / Phase 3 Locked** | **APPROVED** |
| `partitioning_unit` | Contract-level (0 contract leakage) | **SRD-mandated** | **APPROVED** |
| `tokenizer_backbone` | `nlpaueb/legal-bert-base-uncased` | **Engineering Decision** | **APPROVED IN PRINCIPLE** |
| `max_seq_length` | 512 tokens | **Engineering Decision** | **APPROVED IN PRINCIPLE** |
| `doc_stride` | 256 tokens (50% window overlap) | **Data-Dependent Decision** | **APPROVED IN PRINCIPLE** |
| `chunk_id_format` | `{contract_id}__chunk_{chunk_index:04d}` | **Engineering Decision** | **APPROVED IN PRINCIPLE** |
| `multi_span_policy` | Preserve all gold spans per chunk (zero collapse) | **Engineering Decision** | **APPROVED IN PRINCIPLE** |
| `negative_sampling_policy` | Option B ($K=4$ negative chunks per contract) | **Data-Dependent Decision** | **PROPOSED FOR LOCK** |
| `sampling_seed` | 42 | **Engineering Decision** | **PROPOSED FOR LOCK** |
| `no_answer_target` | `start_token = 0, end_token = 0` (CLS token) | **Engineering Decision** | **PROPOSED FOR LOCK** |

---

## 7. Explicit List of Decisions Requiring Final Human Approval

1. **Negative Sampling Option & $K$ Value:** Approval of Option B ($K=4$ negative chunks per contract) based on the multi-task chunk distribution and $81.2\%$ throughput reduction.
2. **Static Preprocessed Artifact Storage:** Approval to generate `data/processed/train_chunks.json`, `val_chunks.json`, and `test_chunks.json` upon lock confirmation in Phase 4C.

---

## 8. Generated Processed Dataset Artifacts & Final Statistics (Phase 4C)

The Phase 4C preprocessing pipeline execution successfully generated the following locked dataset artifacts in `data/processed/`:

- **Train Split (`data/processed/train_chunks.json`):** **2,738 chunks** (357 contracts, Option B $K=4$ sampled negatives).
  - **Union-Positive Chunks:** 1,380 chunks
  - **Sampled Union-Negative Chunks:** 1,358 chunks
  - **Negative Deficit Note:** 34 contracts contain fewer than 4 eligible all-negative chunks; all available negatives are retained for these contracts (cumulative 70-chunk deficit from $357 \times 4 = 1,428$ max potential negatives $\rightarrow$ 1,358 retained negatives).
- **Validation Split (`data/processed/val_chunks.json`):** **3,193 chunks** (76 contracts, 100% sliding window retention, 0 downsampling).
- **Test Split (`data/processed/test_chunks.json`):** **3,178 chunks** (77 contracts, 100% sliding window retention, 0 downsampling).
- **Total Processed Dataset Size:** 9,109 chunks across all splits.

### Phase 4B.1 Estimate vs Phase 4C Implementation Clarification
- **Phase 4B.1 Preliminary Estimate (2,745 chunks):** Sliced text windows at 512 text tokens without reserving slots for `[CLS]` and `[SEP]` special tokens. This produced 1,387 union-positive chunk assignments + 1,358 sampled negatives = 2,745 chunks.
- **Phase 4C Final Implementation (2,738 chunks):** Correctly allocates 2 slots for `[CLS]` (token 0) and `[SEP]` (token $N-1$), setting `max_window_text = 510` text tokens per window so `len(input_ids) <= 512`. This eliminated 7 duplicate positive-window overlaps across adjacent 256-stride windows while preserving 100% of gold spans in overlapping windows, yielding 1,380 union-positive chunks + 1,358 sampled negatives = **2,738 total Train chunks**.

- **Contract Leakage Gate:** PASSED (0 contract overlap across partitions).
- **Test Suite Status:** 34 unit tests PASSED cleanly.


