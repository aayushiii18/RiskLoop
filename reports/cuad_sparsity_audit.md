# RiskLoop Phase 2 — CUAD v1 Dataset Inspection & Sparsity Audit Report
**Date:** August 30, 2026  
**Dataset Version:** CUAD v1 (`aok_v1.0`)  
**Total Audited Contracts:** 510  
**Requirements Baseline:** RiskLoop SRD v1.1 (FR-01 through FR-13)

---

## 1. Executive Summary
A comprehensive pre-experiment data-sparsity audit was conducted across all 41 task categories present in `data/raw/CUAD_v1.json`.
- **Total Candidate Tasks Identified:** 41
- **Potentially Viable Tasks (Pre-Split Feasibility >= 41 Positive Contracts):** 33
- **Failing Tasks (< 41 Positive Contracts):** 8

> [!NOTE]
> **Pre-Split Feasibility Screen Notice:** Having >= 41 total positive contracts across the 510 contracts is a necessary pre-split feasibility screen for satisfying Train >= 25, Val >= 8, Test >= 8. Final adequacy must be verified post-splitting once the contract split map is generated.

> [!IMPORTANT]
> **Scientific Control Notice:** Per SRD FR-02, no task definitions have been invented, and the final viable task set is presented as evidence for review prior to freezing the task configuration.

---

## 2. Dataset Schema & Structure
- **Root Structure:** JSON dictionary containing `version` (`aok_v1.0`) and `data` (list of 510 contracts).
- **Contract Record:** Each contract element contains a `title` string and a `paragraphs` list.
- **Context Representation:** `paragraphs[0]['context']` contains the full raw text of the contract.
- **QAs Array:** `paragraphs[0]['qas']` contains 41 question objects.
- **QA Schema:**
  - `id`: `{contract_title}__{category_name}`
  - `question`: Full natural language question prompt.
  - `is_impossible`: Boolean (`True` if negative/no answer, `False` if positive span exists).
  - `answers`: List of `{'text': str, 'answer_start': int}` objects.

---

## 3. Complete Per-Task Sparsity Audit Table

| Task Name | Pos Contracts | Prevalence | Pos Spans | Mean Spans/Pos Cntr | Max Spans | Top 10% Share | Top 20% Share | Pre-Split Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Affiliate License-Licensee` | 59/510 | 11.6% | 115 | 1.9 | 10 | 31.3% | 47.0% | **PASS** |
| `Affiliate License-Licensor` | 23/510 | 4.5% | 69 | 3.0 | 10 | 40.6% | 52.2% | FAIL |
| `Agreement Date` | 470/510 | 92.2% | 476 | 1.0 | 2 | 11.1% | 21.0% | **PASS** |
| `Anti-Assignment` | 374/510 | 73.3% | 654 | 1.7 | 10 | 25.1% | 41.6% | **PASS** |
| `Audit Rights` | 214/510 | 42.0% | 643 | 3.0 | 19 | 33.6% | 50.5% | **PASS** |
| `Cap On Liability` | 275/510 | 53.9% | 672 | 2.4 | 16 | 29.0% | 45.4% | **PASS** |
| `Change Of Control` | 121/510 | 23.7% | 253 | 2.1 | 11 | 27.3% | 42.3% | **PASS** |
| `Competitive Restriction Exception` | 76/510 | 14.9% | 125 | 1.6 | 6 | 24.8% | 40.8% | **PASS** |
| `Covenant Not To Sue` | 100/510 | 19.6% | 173 | 1.7 | 5 | 24.3% | 41.0% | **PASS** |
| `Document Name` | 510/510 | 100.0% | 521 | 1.0 | 2 | 11.9% | 21.7% | **PASS** |
| `Effective Date` | 390/510 | 76.5% | 447 | 1.1 | 3 | 18.1% | 30.2% | **PASS** |
| `Exclusivity` | 180/510 | 35.3% | 410 | 2.3 | 12 | 27.1% | 42.0% | **PASS** |
| `Expiration Date` | 413/510 | 81.0% | 467 | 1.1 | 7 | 19.9% | 29.3% | **PASS** |
| `Governing Law` | 437/510 | 85.7% | 464 | 1.1 | 4 | 15.3% | 24.8% | **PASS** |
| `Insurance` | 166/510 | 32.5% | 560 | 3.4 | 17 | 35.9% | 52.7% | **PASS** |
| `Ip Ownership Assignment` | 124/510 | 24.3% | 318 | 2.6 | 15 | 27.0% | 43.1% | **PASS** |
| `Irrevocable Or Perpetual License` | 70/510 | 13.7% | 165 | 2.4 | 10 | 30.9% | 49.7% | **PASS** |
| `Joint Ip Ownership` | 46/510 | 9.0% | 115 | 2.5 | 11 | 30.4% | 52.2% | **PASS** |
| `License Grant` | 255/510 | 50.0% | 777 | 3.0 | 16 | 30.9% | 49.7% | **PASS** |
| `Liquidated Damages` | 61/510 | 12.0% | 121 | 2.0 | 10 | 34.7% | 48.8% | **PASS** |
| `Minimum Commitment` | 165/510 | 32.4% | 424 | 2.6 | 14 | 30.7% | 47.4% | **PASS** |
| `Most Favored Nation` | 28/510 | 5.5% | 38 | 1.4 | 3 | 23.7% | 39.5% | FAIL |
| `No-Solicit Of Customers` | 34/510 | 6.7% | 58 | 1.7 | 6 | 25.9% | 36.2% | FAIL |
| `No-Solicit Of Employees` | 59/510 | 11.6% | 91 | 1.5 | 4 | 20.9% | 36.3% | **PASS** |
| `Non-Compete` | 119/510 | 23.3% | 259 | 2.2 | 12 | 32.0% | 46.7% | **PASS** |
| `Non-Disparagement` | 38/510 | 7.5% | 65 | 1.7 | 3 | 18.5% | 32.3% | FAIL |
| `Non-Transferable License` | 138/510 | 27.1% | 298 | 2.2 | 12 | 30.5% | 47.0% | **PASS** |
| `Notice Period To Terminate Renewal` | 111/510 | 21.8% | 122 | 1.1 | 3 | 18.9% | 27.9% | **PASS** |
| `Parties` | 509/510 | 99.8% | 2554 | 5.0 | 55 | 22.0% | 33.9% | **PASS** |
| `Post-Termination Services` | 182/510 | 35.7% | 450 | 2.5 | 23 | 36.7% | 52.0% | **PASS** |
| `Price Restrictions` | 15/510 | 2.9% | 27 | 1.8 | 7 | 37.0% | 48.1% | FAIL |
| `Renewal Term` | 176/510 | 34.5% | 210 | 1.2 | 5 | 21.4% | 33.3% | **PASS** |
| `Revenue/Profit Sharing` | 166/510 | 32.5% | 418 | 2.5 | 11 | 29.2% | 47.1% | **PASS** |
| `Rofr/Rofo/Rofn` | 85/510 | 16.7% | 367 | 4.3 | 22 | 37.3% | 56.7% | **PASS** |
| `Source Code Escrow` | 13/510 | 2.5% | 66 | 5.1 | 11 | 31.8% | 42.4% | FAIL |
| `Termination For Convenience` | 183/510 | 35.9% | 246 | 1.3 | 4 | 20.7% | 35.4% | **PASS** |
| `Third Party Beneficiary` | 32/510 | 6.3% | 39 | 1.2 | 4 | 28.2% | 35.9% | FAIL |
| `Uncapped Liability` | 111/510 | 21.8% | 167 | 1.5 | 5 | 25.7% | 39.5% | **PASS** |
| `Unlimited/All-You-Can-Eat-License` | 17/510 | 3.3% | 32 | 1.9 | 5 | 31.2% | 53.1% | FAIL |
| `Volume Restriction` | 82/510 | 16.1% | 171 | 2.1 | 10 | 28.7% | 45.6% | **PASS** |
| `Warranty Duration` | 75/510 | 14.7% | 176 | 2.3 | 10 | 30.7% | 47.7% | **PASS** |

---

## 4. Potentially Viable Tasks (33 Tasks)
The following 33 tasks meet or exceed the global minimum adequacy threshold of 41 positive contracts (enforcing Train >= 25, Val >= 8, Test >= 8):

- `Affiliate License-Licensee` (59 positive contracts, 115 spans)
- `Agreement Date` (470 positive contracts, 476 spans)
- `Anti-Assignment` (374 positive contracts, 654 spans)
- `Audit Rights` (214 positive contracts, 643 spans)
- `Cap On Liability` (275 positive contracts, 672 spans)
- `Change Of Control` (121 positive contracts, 253 spans)
- `Competitive Restriction Exception` (76 positive contracts, 125 spans)
- `Covenant Not To Sue` (100 positive contracts, 173 spans)
- `Document Name` (510 positive contracts, 521 spans)
- `Effective Date` (390 positive contracts, 447 spans)
- `Exclusivity` (180 positive contracts, 410 spans)
- `Expiration Date` (413 positive contracts, 467 spans)
- `Governing Law` (437 positive contracts, 464 spans)
- `Insurance` (166 positive contracts, 560 spans)
- `Ip Ownership Assignment` (124 positive contracts, 318 spans)
- `Irrevocable Or Perpetual License` (70 positive contracts, 165 spans)
- `Joint Ip Ownership` (46 positive contracts, 115 spans)
- `License Grant` (255 positive contracts, 777 spans)
- `Liquidated Damages` (61 positive contracts, 121 spans)
- `Minimum Commitment` (165 positive contracts, 424 spans)
- `No-Solicit Of Employees` (59 positive contracts, 91 spans)
- `Non-Compete` (119 positive contracts, 259 spans)
- `Non-Transferable License` (138 positive contracts, 298 spans)
- `Notice Period To Terminate Renewal` (111 positive contracts, 122 spans)
- `Parties` (509 positive contracts, 2554 spans)
- `Post-Termination Services` (182 positive contracts, 450 spans)
- `Renewal Term` (176 positive contracts, 210 spans)
- `Revenue/Profit Sharing` (166 positive contracts, 418 spans)
- `Rofr/Rofo/Rofn` (85 positive contracts, 367 spans)
- `Termination For Convenience` (183 positive contracts, 246 spans)
- `Uncapped Liability` (111 positive contracts, 167 spans)
- `Volume Restriction` (82 positive contracts, 171 spans)
- `Warranty Duration` (75 positive contracts, 176 spans)

---

## 5. Failing Tasks (8 Tasks)
The following 8 tasks fail the SRD data-adequacy gate and cannot support the controlled experiment:

- `Affiliate License-Licensor`: Only 23 positive contracts (requires >= 41).
- `Most Favored Nation`: Only 28 positive contracts (requires >= 41).
- `No-Solicit Of Customers`: Only 34 positive contracts (requires >= 41).
- `Non-Disparagement`: Only 38 positive contracts (requires >= 41).
- `Price Restrictions`: Only 15 positive contracts (requires >= 41).
- `Source Code Escrow`: Only 13 positive contracts (requires >= 41).
- `Third Party Beneficiary`: Only 32 positive contracts (requires >= 41).
- `Unlimited/All-You-Can-Eat-License`: Only 17 positive contracts (requires >= 41).

---

## 6. Contract Concentration Diagnostics
High concentration indicates that a small fraction of contracts account for a large portion of positive spans:
- **Highest Span Concentration Tasks:**
  - `Rofr/Rofo/Rofn`: Top 20% of positive contracts account for 56.7% of all positive spans.
  - `Post-Termination Services`: Top 20% of positive contracts account for 51.3% of all positive spans.
  - `Insurance`: Top 20% of positive contracts account for 51.8% of all positive spans.
  - `Parties`: Highest maximum spans in a single contract (55 spans).

---

## 7. TBD Decisions Register
1. **Final Viable Task Selection:** Selection of candidate tasks for the 12-run experiment must be formally locked in Phase 3.
2. **Tokenizer & Sequence Length:** Pending chunking and sequence length optimization.
3. **Chunk Stride & Span Alignment:** Pending span length distribution audit.
4. **Primary Evaluation Metrics:** To be selected post-lock.