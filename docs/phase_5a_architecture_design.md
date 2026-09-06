# RiskLoop — Phase 5A Architecture Design & Pre-Training Freeze Document

**Date:** September 6, 2026  
**Status:** FROZEN AND VERIFIED (Phase 5A Complete)  
**Requirements Baseline:** RiskLoop SRD v1.1 (FR-15, FR-21..FR-25, FR-29..FR-37, NFR-01..NFR-08, NFR-13)  
**Baseline Commit:** `98b8f50` (`feat: implement deterministic CUAD preprocessing pipeline`)

---

## 1. Model Architecture Specification

### Condition A: Independent Single-Task Baseline Models (`SingleTaskModel`)
- **Backbone:** `nlpaueb/legal-bert-base-uncased` (12 layers, 768 hidden size, 12 attention heads).
- **Classification Head:** Linear layer `nn.Linear(768, 2)` mapping text token representations to start and end logits.
- **Head Initialization:**
  $$\text{weight} \sim \text{Normal}(\text{mean}=0.0, \text{std}=0.02)$$
  $$\text{bias} = 0$$
- **Model Counts & Isolation:** 3 separate, fully independent models trained on seeds $\{42, 43, 44\}$ for each of the 3 selected tasks ($3 \times 3 = 9$ runs total). Total parameter count per model: **109,483,778 parameters** ($109,482,240 \text{ encoder} + 1,538 \text{ head}$).

### Condition B: Shared Multi-Task Model (`MultiTaskModel`)
- **Backbone:** 1 shared `nlpaueb/legal-bert-base-uncased` encoder.
- **Classification Heads:** 3 task-specific linear heads `nn.Linear(768, 2)` for `Cap On Liability`, `Anti-Assignment`, and `Termination For Convenience`.
- **Head Initialization:**
  $$\text{weight} \sim \text{Normal}(\text{mean}=0.0, \text{std}=0.02)$$
  $$\text{bias} = 0$$
- **Model Counts & Isolation:** 1 shared multi-task model trained on seeds $\{42, 43, 44\}$ ($1 \times 3 = 3$ runs total). Total parameter count for the joint model: **109,486,854 parameters** ($109,482,240 \text{ encoder} + 4,614 \text{ 3 heads}$).

### Parameter Count Reconciliation
- **Encoder Parameter Count:** $109,482,240$ parameters.
- **Discrepancy Explanation:** The earlier preliminary manual text calculation ($109,480,704$) omitted the token-type embedding parameters (`embeddings.token_type_embeddings.weight`: $2 \times 768 = \mathbf{1,536}$ parameters). Adding $1,536$ to $109,480,704$ yields the exact instantiated PyTorch encoder count of $109,482,240$.
- **Authoritative PyTorch Counts:**
  - `SingleTaskModel` = **109,483,778** parameters ($109,482,240 + 1,538$).
  - `MultiTaskModel` = **109,486,854** parameters ($109,482,240 + 4,614$).

---

## 2. Full Evaluation Pipeline & Primary Metric Specification

### Primary Validation Metric Definition
The Primary Validation Metric for epoch selection, adoption protocol evaluations (Stage 2A/2B), and representative model selection is **Positive-Class F1 (`class_1_f1`)**, evaluated at the chunk level across validation chunks.

> **"Primary metric measures binary clause-presence detection at chunk level; it does not by itself measure correctness of the extracted span boundaries."**

### End-to-End Logit-to-Metric Decoding Pipeline
1. **Model Logits:** For a chunk record with $512$ tokens, the model outputs `start_logits` $(B, 512)$ and `end_logits` $(B, 512)$.
2. **Logit Masking:** For any token position $i$ where `attention_mask[i] == 0` (padding), set:
   $$\text{start\_logits}[i] = -10000.0, \quad \text{end\_logits}[i] = -10000.0$$
3. **No-Answer Score:** Token position $0$ corresponds to `[CLS]`:
   $$\text{no\_answer\_score} = \text{start\_logits}[0] + \text{end\_logits}[0]$$
4. **Max Span Score:** Searching over valid text token spans $1 \le i \le j \le 510$ with span length $j - i \le 256$:
   $$\text{max\_span\_score} = \max_{1 \le i \le j \le 510, \, j-i \le 256} (\text{start\_logits}[i] + \text{end\_logits}[j])$$
5. **Raw Model Score:**
   $$\text{model\_score} = \text{max\_span\_score} - \text{no\_answer\_score}$$
6. **Binary Prediction Classification:**
   A chunk is classified as **Predicted Positive (1)** iff $\text{model\_score} \ge T$ (where default decision threshold $T = 0.0$); otherwise **Predicted Negative / No-Answer (0)**.
7. **Ground Truth Label Assignment:**
   A chunk's ground truth label is $y_{\text{true}} = 1$ if `task_targets[task]["has_positive"] == True` (containing $\ge 1$ complete gold span inside the chunk window); otherwise $y_{\text{true}} = 0$.
8. **Confusion Matrix & Positive-Class F1 Calculation:**
   Across all $N_{\text{val}}$ validation chunks ($3,193$ chunks):
   - $TP$: Chunks where $y_{\text{true}} = 1$ and $y_{\text{pred}} = 1$.
   - $FP$: Chunks where $y_{\text{true}} = 0$ and $y_{\text{pred}} = 1$.
   - $FN$: Chunks where $y_{\text{true}} = 1$ and $y_{\text{pred}} = 0$.
   - $TN$: Chunks where $y_{\text{true}} = 0$ and $y_{\text{pred}} = 0$.
   $$\text{Precision}_1 = \frac{TP}{TP + FP}, \quad \text{Recall}_1 = \frac{TP}{TP + FN}$$
   $$\mathbf{class\_1\_f1} = \frac{2 \cdot \text{Precision}_1 \cdot \text{Recall}_1}{\text{Precision}_1 + \text{Recall}_1}$$
   - **Zero-Division Behavior:** If $TP + FP = 0$ or $TP + FN = 0$, $\text{Precision}_1 = 0.0$, $\text{Recall}_1 = 0.0$, and $\mathbf{class\_1\_f1} = 0.0$.

---

## 3. Multi-Span Training & Evaluation Consistency

- **Training Target Selection:** `CUADChunkDataset` uses `multi_span_option="first_span"`, selecting `spans[0]` (first gold span in character order) as the single `(start_token, end_token)` pair for CrossEntropy loss calculation.
- **Evaluation Preservation:** **100% of gold spans** stored in preprocessed JSON artifacts (`task_targets[task]["spans"]`) are preserved without dropping. During validation and test evaluations, all gold spans remain available for contract-level span-matching metrics.
- **Chunk Labeling Consistency:** For chunk-level binary classification, `has_positive = True` if **any** gold span is fully contained in the chunk window, preserving multi-span positive status regardless of which span is selected for single-span CrossEntropy target supervision.

---

## 4. Pre-Training Configuration Lock Parameters

| Parameter Item | Value | Status |
| :--- | :--- | :---: |
| **Model Backbone** | `nlpaueb/legal-bert-base-uncased` | **LOCKED** |
| **Max Sequence Length** | 512 tokens (510 text + CLS + SEP) | **LOCKED** |
| **Sliding Window Stride** | 256 tokens | **LOCKED** |
| **Negative Sampling Policy** | Option B ($K=4$, seed 42) | **LOCKED** |
| **Selected Tasks** | Cap On Liability, Anti-Assignment, Termination For Convenience | **LOCKED** |
| **Train Chunks** | 2,738 chunks | **LOCKED** |
| **Experimental Matrix** | 12 runs (Condition A: 9, Condition B: 3) | **LOCKED** |
| **Seeds (3)** | `{42, 43, 44}` | **LOCKED** |
| **Linear Head Initialization** | `weight ~ Normal(mean=0.0, std=0.02)`, `bias = 0` | **LOCKED** |
| **Multi-Span Training Target** | First-Span (`spans[0]`) | **LOCKED** |
| **Optimizer** | AdamW (weight decay = 0.01, linear warmup ratio = 0.10) | **LOCKED** |
| **Learning Rate** | `3e-5` | **LOCKED** |
| **Epoch Count** | 4 epochs | **LOCKED** |
| **Effective Batch Size** | 16 (`batch_size` 8 $\times$ 2 grad accum) | **LOCKED** |
| **AMP Policy** | Consistent FP16 on CUDA; FP32 on CPU | **LOCKED** |
| **Primary Validation Metric** | Positive-Class F1 (`class_1_f1`), chunk-level | **LOCKED** |
| **Checkpoint Selection Rule** | Max validation `class_1_f1` (tie-breaker: earlier epoch) | **LOCKED** |
| **Test Set Isolation** | Touched exactly once post-freeze by `TestSetEvaluator` | **LOCKED** |
