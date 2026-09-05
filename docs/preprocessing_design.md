# RiskLoop — Preprocessing Design & Analysis Document

**Date:** September 5, 2026  
**Status:** IMPLEMENTED AND VERIFIED (Phase 4C Complete)  
**Requirements Baseline:** RiskLoop SRD v1.1  
**Locked Inputs:**
- Selected Task Set: `Cap On Liability`, `Anti-Assignment`, `Termination For Convenience`
- Canonical Split Map: [`data/splits/contract_split_map.json`](file:///c:/Users/aayushi/OneDrive/Documents/RiskLoop/data/splits/contract_split_map.json) (Seed 42, 357 Train / 76 Val / 77 Test)

---

## 1. Executive Summary

This document establishes the technical design, architectural requirements, and implementation details for the RiskLoop dataset preprocessing, tokenization, and sliding-window chunking pipeline.

The preprocessing pipeline has been fully implemented in `src/riskloop/data/preprocessing.py` and validated with 34 passing unit tests.

---

## 2. Empirical CUAD Annotation & Document Length Observations

Direct inspection of all 510 contracts in `data/raw/CUAD_v1.json` yielded the following empirical findings for the three locked tasks:

### Contract Text Length Distribution (Characters)
- **Total Audited Contracts:** 510
- **Minimum Length:** 645 characters (*Shortest contract: `RMRGROUPINC...`*)
- **25th Percentile:** 16,416 characters (~4,100 words)
- **Median Length:** 33,143 characters (~8,300 words)
- **Mean Length:** 52,563.0 characters (~13,100 words)
- **75th Percentile:** 66,394 characters (~16,600 words)
- **90th Percentile:** 122,117 characters (~30,500 words)
- **Maximum Length:** 338,211 characters (~84,500 words) (*Longest contract: `MANUFACTURERSSERVICESLTD...`*)

> [!IMPORTANT]
> **Implication for Model Input Length:**  
> Standard Transformer models accept maximum sequence lengths of 512 tokens (~2,000 characters). 509 out of 510 contracts (99.8%) exceed 512 tokens (the shortest contract `RMRGROUPINC...` is 645 characters / ~160 tokens; the median contract is 33,143 characters / ~10,000 tokens, which is $\sim 20\times$ larger than the max model window). Therefore, **sliding-window chunking is strictly mandatory**.

### Annotation Quality & Offset Matching Findings (1,572 Total Positive Spans Checked)
1. **Character Slice Precision:** **0 exact-match failures** across all 1,572 positive spans. `context[answer_start : answer_start + len(text)]` matches `text` with 100% character accuracy.
2. **Multi-Answer Contract Frequency:**
   - `Cap On Liability`: 161 of 275 positive contracts contain multiple annotated answer spans.
   - `Anti-Assignment`: 158 of 374 positive contracts contain multiple annotated answer spans.
   - `Termination For Convenience`: 50 of 183 positive contracts contain multiple annotated answer spans.
3. **Overlapping Span Annotations:**
   - `Cap On Liability`: 8 contracts contain partially overlapping span annotations.
   - `Anti-Assignment`: 4 contracts contain partially overlapping span annotations.
   - `Termination For Convenience`: 1 contract contains partially overlapping span annotations.
4. **Relative Span Position in Document Context (0.0 = Start, 1.0 = End):**
   - `Cap On Liability`: Median relative position = **0.646** (25th pctile = 0.483, 75th pctile = 0.780). Spans appear primarily in middle-to-latter liability sections.
   - `Anti-Assignment`: Median relative position = **0.780** (25th pctile = 0.622, 75th pctile = 0.874). Spans appear predominantly in boilerplate/miscellaneous sections near the end.
   - `Termination For Convenience`: Median relative position = **0.593** (25th pctile = 0.358, 75th pctile = 0.739). Spans appear mostly in term and termination clauses in the middle portion of contracts.

---

## 3. SRD Preprocessing Requirements Matrix

The SRD preprocessing requirements are categorized into explicit, data-dependent (TBD), and unspecified requirements:

| Requirement Category | Requirement Item | Status | SRD Traceability / Rule |
| :--- | :--- | :---: | :--- |
| **Explicitly Specified** | Contract-Level Partitioning | **LOCKED** | FR-05, FR-06, FR-07 (0 contract leakage, 70/15/15 ratio) |
| **Explicitly Specified** | Extractive Span Prediction Formulation | **SPECIFIED** | $P(\text{start}=i), P(\text{end}=j)$ classification per task head |
| **Explicitly Specified** | Single-Pass Test Evaluation | **LOCKED** | FR-38, FR-39, FR-40 |
| **Data-Dependent** | Tokenizer / Backbone Model Family | **TBD** | Requires selection among RoBERTa / Legal-BERT / DeBERTa |
| **Data-Dependent** | Maximum Sequence Length | **PROPOSED** | 512 tokens (standard transformer limit) |
| **Data-Dependent** | Sliding Window Stride / Overlap | **PROPOSED** | Stride = 256 tokens (50% overlap / 256 token stride) |
| **Data-Dependent** | Negative Sampling / No-Answer Ratio | **TBD** | Requires lock on negative chunk ratio per contract |
| **Unspecified** | Specific HuggingFace Checkpoint Name | **TBD** | e.g. `roberta-base` vs `nlpaueb/legal-bert-base-uncased` |

---

## 4. Tokenizer & Backbone Candidates Evaluation

Three primary HuggingFace Transformer model families fit extractive QA on legal text:

### Candidate 1: `nlpaueb/legal-bert-base-uncased` (Legal-BERT)
- **Architecture:** BERT-base WordPiece tokenizer (512 max position embeddings).
- **Domain Fit:** Pre-trained on 12 GB of legal text (SEC filings, court cases, contracts).
- **Pros:** Domain-specific vocabulary handles legal jargon ("indemnification", "sublicensor", "counterparty") with fewer subword splits.
- **Cons:** Base size (110M params); lower general capacity compared to RoBERTa-large.

### Candidate 2: `roberta-base` / `roberta-large` (RoBERTa)
- **Architecture:** Byte-level BPE tokenizer (512 max position embeddings).
- **Domain Fit:** General-domain LM trained on 160 GB text. Standard benchmark baseline for QA.
- **Pros:** Robust representation power; fast tokenizer offset mapping (`return_offsets_mapping=True`).
- **Cons:** General vocabulary splits complex legal terms into multiple subwords.

### Candidate 3: `microsoft/deberta-v3-base` (DeBERTa-v3)
- **Architecture:** Disentangled attention with SPM tokenizer (512 max position embeddings).
- **Domain Fit:** Advanced disentangled attention mechanism.
- **Pros:** SOTA performance on MRC / extractive QA benchmarks.
- **Cons:** Slightly higher computational latency during training.

---

## 5. Sliding-Window Chunking Strategy

To process long contracts (median 33,143 characters / ~10,000 tokens) without losing span context at window boundaries:

### Proposed Strategy: Overlapping Sliding Window (Window = 512, Stride = 256)
- **Window Size ($L_{max}$):** 512 tokens.
- **Stride ($S$):** 256 tokens (50% overlap).
- **Rationale:** 
  - Every 512-token window overlaps by 256 tokens with the adjacent window.
  - Ensures that any positive answer span up to 256 tokens in length will fall **100% completely inside at least one window chunk**, avoiding severed span boundaries.
- **Chunk Expansion Factor:** Generates $\sim 35 - 45$ chunks per contract, producing $\sim 18,000$ total chunks across the 510 contracts.

---

## 6. Character-to-Token Span Alignment Design

Mapping raw character offsets `(char_start, char_end)` to token indices `(token_start, token_end)` within a 512-token chunk:

1. **Fast Tokenizer Offset Mapping:** Use `FastTokenizer` with `return_offsets_mapping=True` to get `offset_mapping[i] = (start_char, end_char)` for each token $i$.
2. **Subword Boundary Expansion:**
   - `token_start`: Index of the token whose character range contains `char_start`.
   - `token_end`: Index of the token whose character range contains `char_end - 1`.
3. **Chunk Containment Check:**
   - A span is classified as **Positive** in Chunk $C_k$ iff the entire span `[char_start, char_end]` is fully contained within Chunk $C_k$'s character range.
   - If a span is only partially contained in Chunk $C_k$, it is treated as **No-Answer** for Chunk $C_k$ to prevent truncated target supervision.
4. **No-Answer Classification Target:**
   - Chunks containing no valid positive span for task $t$ set `start_position = 0` and `end_position = 0` (pointing to the special `[CLS]` / `<s>` token).

---

## 7. Negative Sampling & No-Answer Handling (TBD)

- **Class Imbalance Problem:** Out of $\sim 40$ chunks generated for a contract, typically only $1-3$ chunks contain a positive span for a given task. The remaining $\sim 37$ chunks are negative (no-answer). Training on all negative chunks would yield a $\sim 95\%$ negative label imbalance.
- **TBD Policy Decision:** Phase 4B must decide whether to:
  - Option A: Keep all negative chunks (full contract coverage).
  - Option B: Downsample negative chunks (e.g., keep all positive chunks + 4 randomly sampled negative chunks per contract).

---

## 8. Leakage Invariants & Reproducibility Requirements

1. **Contract Partition Invariant:**
   $$\text{Contract } C_i \in \text{Partition } P \implies \forall \text{ Chunk } K \in C_i, K \in \text{Partition } P$$
   All chunks derived from contract $C_i$ inherit $C_i$'s partition assignment from [`data/splits/contract_split_map.json`](file:///c:/Users/aayushi/OneDrive/Documents/RiskLoop/data/splits/contract_split_map.json). Zero cross-partition chunk leakage.
2. **Deterministic Chunk Identifier:**
   $$\text{chunk\_id} = \text{f"{contract\_id}\_\_chunk\_\{chunk\_index:04d\}"}$$

---

## 9. Proposed Implementation Stages for Phase 4

1. **Phase 4B (Preprocessing Configuration Lock):** Lock tokenizer family, max sequence length (512), stride (256), and negative sampling policy in a versioned config artifact.
2. **Phase 4C (Preprocessing Pipeline Implementation):** Implement `src/riskloop/data/preprocessing.py` to produce chunk dataset files (`data/processed/train_chunks.json`, `val_chunks.json`, `test_chunks.json`).
3. **Phase 4D (Validation & Leakage Verification):** Verify 0 chunk leakage, 100% token offset alignment, and pytest suite execution.

---

## 10. Open Decisions Register (Requiring User Review)

1. **Tokenizer / Backbone Selection:** Approval of `Legal-BERT` vs `RoBERTa-base` vs `DeBERTa-v3-base`.
2. **Sliding Window Overlap Stride:** Approval of 512 window / 256 stride.
3. **Negative Sampling Ratio:** Selection of Option A (full negative chunks) vs Option B (downsampled negative chunks).

---

## 11. Implemented Preprocessing Pipeline & Final Dataset Metrics (Phase 4C)

The Phase 4C pipeline implemented in `src/riskloop/data/preprocessing.py` has generated the following finalized dataset splits:

- **Train Split (`data/processed/train_chunks.json`):** **2,738 chunks** (357 contracts, Option B $K=4$ negative sampling, 1,380 union-positive chunks, 1,358 sampled negatives).
- **Validation Split (`data/processed/val_chunks.json`):** **3,193 chunks** (76 contracts, 100% chunk retention).
- **Test Split (`data/processed/test_chunks.json`):** **3,178 chunks** (77 contracts, 100% chunk retention).
- **Total Dataset Size:** 9,109 chunks across all splits.
- **Verification:** 34 pytest unit tests pass cleanly. Zero contract leakage across partitions.

