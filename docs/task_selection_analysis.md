# RiskLoop — Task Selection Analysis Document

**Date:** September 5, 2026  
**Baseline Dataset:** CUAD v1 (`aok_v1.0`, 510 commercial contracts)  
**Requirements Baseline:** RiskLoop SRD v1.1 (FR-01 through FR-13)  
**Status:** **APPROVED & LOCKED** (Candidate Set A selected and post-split verified)

---

## 1. Executive Summary & Purpose

This document provides the evidence-based Task Selection Analysis for the **RiskLoop** multi-task contract-risk machine learning system.

Following user review and post-split verification against the SRD's data-adequacy gates (Train $\ge 25$, Val $\ge 8$, Test $\ge 8$ positive contracts), **Candidate Set A** has been selected and locked for the controlled 12-run experiment:
1. `Cap On Liability` (Liability / exposure risk)
2. `Anti-Assignment` (Assignment / transfer restriction risk)
3. `Termination For Convenience` (Termination / exit-right risk)

Per SRD FR-02 and Phase 3 guidelines:
- **No task definitions were invented.**
- **No model preprocessing, tokenization, chunking, or model training has been performed.**
- **Contracts are the sole independent partitioning unit, guaranteeing zero split leakage.**

---

## 2. Complete Candidate Inventory (All 41 CUAD Categories)

Below is the complete inventory of all 41 CUAD category annotations extracted directly from `data/raw/CUAD_v1.json` during the Phase 2 audit:

| Task / Category Name | Positive Contracts | Prevalence | Total Positive Spans | Mean Spans / Pos Contract | Max Spans in Single Contract | Top 10% Span Share | Top 20% Span Share | Pre-Split Feasibility Status ($\ge 41$ Pos Contracts) | Sparsity / Concentration Observations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Affiliate License-Licensee` | 59/510 | 11.6% | 115 | 1.9 | 10 | 31.3% | 47.0% | **PASS** | Sparse; localized licensee affiliate grants. |
| `Affiliate License-Licensor` | 23/510 | 4.5% | 69 | 3.0 | 10 | 40.6% | 52.2% | **FAIL** | Extremely sparse; fails pre-split gate. |
| `Agreement Date` | 470/510 | 92.2% | 476 | 1.0 | 2 | 11.1% | 21.0% | **PASS** | High prevalence; strictly single-span metadata. |
| `Anti-Assignment` | 374/510 | 73.3% | 654 | 1.7 | 10 | 25.1% | 41.6% | **PASS** | High prevalence; transfer restriction risk. |
| `Audit Rights` | 214/510 | 42.0% | 643 | 3.0 | 19 | 33.6% | 50.5% | **PASS** | Moderate prevalence; multi-span compliance/inspection. |
| `Cap On Liability` | 275/510 | 53.9% | 672 | 2.4 | 16 | 29.0% | 45.4% | **PASS** | High prevalence; liability exposure risk. |
| `Change Of Control` | 121/510 | 23.7% | 253 | 2.1 | 11 | 27.3% | 42.3% | **PASS** | Moderate prevalence; corporate M&A risk clause. |
| `Competitive Restriction Exception` | 76/510 | 14.9% | 125 | 1.6 | 6 | 24.8% | 40.8% | **PASS** | Low prevalence; carveouts to non-competes. |
| `Covenant Not To Sue` | 100/510 | 19.6% | 173 | 1.7 | 5 | 24.3% | 41.0% | **PASS** | Low-moderate prevalence; IP litigation restriction. |
| `Document Name` | 510/510 | 100.0% | 521 | 1.0 | 2 | 11.9% | 21.7% | **PASS** | 100% prevalence; title metadata string. |
| `Effective Date` | 390/510 | 76.5% | 447 | 1.1 | 3 | 18.1% | 30.2% | **PASS** | High prevalence; contract start date metadata. |
| `Exclusivity` | 180/510 | 35.3% | 410 | 2.3 | 12 | 27.1% | 42.0% | **PASS** | Moderate prevalence; commercial restriction. |
| `Expiration Date` | 413/510 | 81.0% | 467 | 1.1 | 7 | 19.9% | 29.3% | **PASS** | High prevalence; contract termination date metadata. |
| `Governing Law` | 437/510 | 85.7% | 464 | 1.1 | 4 | 15.3% | 24.8% | **PASS** | High prevalence; boilerplate jurisdiction clause. |
| `Insurance` | 166/510 | 32.5% | 560 | 3.4 | 17 | 35.9% | 52.7% | **PASS** | Moderate prevalence; multi-span risk coverage. |
| `Ip Ownership Assignment` | 124/510 | 24.3% | 318 | 2.6 | 15 | 27.0% | 43.1% | **PASS** | Moderate prevalence; core IP transfer clause. |
| `Irrevocable Or Perpetual License` | 70/510 | 13.7% | 165 | 2.4 | 10 | 30.9% | 49.7% | **PASS** | Low prevalence; IP license term clause. |
| `Joint Ip Ownership` | 46/510 | 9.0% | 115 | 2.5 | 11 | 30.4% | 52.2% | **PASS** | Low prevalence (near gate boundary); shared IP. |
| `License Grant` | 255/510 | 50.0% | 777 | 3.0 | 16 | 30.9% | 49.7% | **PASS** | High prevalence; core commercial IP grant. |
| `Liquidated Damages` | 61/510 | 12.0% | 121 | 2.0 | 10 | 34.7% | 48.8% | **PASS** | Low prevalence; breach remedy clause. |
| `Minimum Commitment` | 165/510 | 32.4% | 424 | 2.6 | 14 | 30.7% | 47.4% | **PASS** | Moderate prevalence; financial volume obligation. |
| `Most Favored Nation` | 28/510 | 5.5% | 38 | 1.4 | 3 | 23.7% | 39.5% | **FAIL** | Extremely sparse; fails pre-split gate. |
| `No-Solicit Of Customers` | 34/510 | 6.7% | 58 | 1.7 | 6 | 25.9% | 36.2% | **FAIL** | Extremely sparse; fails pre-split gate. |
| `No-Solicit Of Employees` | 59/510 | 11.6% | 91 | 1.5 | 4 | 20.9% | 36.3% | **PASS** | Low prevalence; non-solicitation clause. |
| `Non-Compete` | 119/510 | 23.3% | 259 | 2.2 | 12 | 32.0% | 46.7% | **PASS** | Moderate prevalence; commercial non-compete. |
| `Non-Disparagement` | 38/510 | 7.5% | 65 | 1.7 | 3 | 18.5% | 32.3% | **FAIL** | Extremely sparse; fails pre-split gate. |
| `Non-Transferable License` | 138/510 | 27.1% | 298 | 2.2 | 12 | 30.5% | 47.0% | **PASS** | Moderate prevalence; license transfer restriction. |
| `Notice Period To Terminate Renewal` | 111/510 | 21.8% | 122 | 1.1 | 3 | 18.9% | 27.9% | **PASS** | Moderate prevalence; termination notice timeline. |
| `Parties` | 509/510 | 99.8% | 2554 | 5.0 | 55 | 22.0% | 33.9% | **PASS** | 99.8% prevalence; high span multiplicity (55 max). |
| `Post-Termination Services` | 182/510 | 35.7% | 450 | 2.5 | 23 | 36.7% | 52.0% | **PASS** | Moderate prevalence; high concentration (52.0% top 20%). |
| `Price Restrictions` | 15/510 | 2.9% | 27 | 1.8 | 7 | 37.0% | 48.1% | **FAIL** | Extremely sparse; fails pre-split gate. |
| `Renewal Term` | 176/510 | 34.5% | 210 | 1.2 | 5 | 21.4% | 33.3% | **PASS** | Moderate prevalence; extension term clause. |
| `Revenue/Profit Sharing` | 166/510 | 32.5% | 418 | 2.5 | 11 | 29.2% | 47.1% | **PASS** | Moderate prevalence; financial revenue split. |
| `Rofr/Rofo/Rofn` | 85/510 | 16.7% | 367 | 4.3 | 22 | 37.3% | 56.7% | **PASS** | Low prevalence; highest top 20% concentration (56.7%). |
| `Source Code Escrow` | 13/510 | 2.5% | 66 | 5.1 | 11 | 31.8% | 42.4% | **FAIL** | Extremely sparse; fails pre-split gate. |
| `Termination For Convenience` | 183/510 | 35.9% | 246 | 1.3 | 4 | 20.7% | 35.4% | **PASS** | Moderate prevalence; exit-right risk. |
| `Third Party Beneficiary` | 32/510 | 6.3% | 39 | 1.2 | 4 | 28.2% | 35.9% | **FAIL** | Extremely sparse; fails pre-split gate. |
| `Uncapped Liability` | 111/510 | 21.8% | 167 | 1.5 | 5 | 25.7% | 39.5% | **PASS** | Moderate prevalence; liability carveout risk. |
| `Unlimited/All-You-Can-Eat-License` | 17/510 | 3.3% | 32 | 1.9 | 5 | 31.2% | 53.1% | **FAIL** | Extremely sparse; fails pre-split gate. |
| `Volume Restriction` | 82/510 | 16.1% | 171 | 2.1 | 10 | 28.7% | 43.9% | **PASS** | Low-moderate prevalence; operational limit. |
| `Warranty Duration` | 75/510 | 14.7% | 176 | 2.3 | 10 | 30.7% | 47.7% | **PASS** | Low-moderate prevalence; warranty term clause. |

---

## 3. Selected Task Set & Risk Rationale

Following candidate evaluation, **Candidate Set A** was selected as the locked task set for the RiskLoop experiment.

### Risk Rationale by Contractual Risk Dimension
- **`Cap On Liability` (Liability / Exposure Risk):** Represents financial risk allocation by capturing monetary recovery limits, liability caps, and claim timeframe restrictions (275 positive contracts / 53.9% prevalence).
- **`Anti-Assignment` (Assignment / Transfer Restriction Risk):** Represents structural contract transfer risk by capturing consent and notice requirements upon contract assignment (374 positive contracts / 73.3% prevalence).
- **`Termination For Convenience` (Termination / Exit-Right Risk):** Represents commercial relationship exit risk by capturing unilateral termination rights without cause and associated notice periods (183 positive contracts / 35.9% prevalence).

---

## 4. Post-Split Sparsity Audit & Gate Verification Results

The deterministic contract split map (`data/splits/contract_split_map.json`) was generated using seed `42` with target ratios 70% Train / 15% Val / 15% Test across all 510 contracts:
- **Train Partition:** 357 contracts (70.0%)
- **Validation Partition:** 76 contracts (14.9%)
- **Test Partition:** 77 contracts (15.1%)

### Post-Split Task Sparsity Results

| Task Name | Contractual Risk Dimension | Train Pos Contracts (Req $\ge 25$) | Val Pos Contracts (Req $\ge 8$) | Test Pos Contracts (Req $\ge 8$) | Train Spans | Val Spans | Test Spans | Post-Split Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Cap On Liability` | Liability / Exposure Risk | **198** | **39** | **38** | 494 | 90 | 88 | **PASS** |
| `Anti-Assignment` | Assignment / Transfer Restriction | **257** | **61** | **56** | 432 | 111 | 111 | **PASS** |
| `Termination For Convenience` | Termination / Exit-Right Risk | **123** | **28** | **32** | 165 | 37 | 44 | **PASS** |

All three selected tasks exceed the post-split SRD thresholds by a substantial margin in every partition.

---

## 5. Critical Scientific Controls & Zero-Leakage Verification

1. **Pre-Split vs. Post-Split Distinction:** The Phase 2 $\ge 41$ positive contract threshold served strictly as a pre-split feasibility filter. Post-split audit confirms that all three tasks pass the mandatory Train $\ge 25$, Val $\ge 8$, and Test $\ge 8$ requirement.
2. **Zero Split Leakage:** Programmatic audit via `audit_split_leakage()` confirmed **0 contract overlap** across Train, Val, and Test partitions.
3. **Partitioning Unit:** Contracts—not answer spans or chunks—are the sole independent partitioning unit. All annotations belonging to contract $C_i$ remain strictly within $C_i$'s assigned split.
