# Multi-Source Business Entity Resolution: Enterprise Record Linkage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harish-k-harish/Business-entity-resolution/blob/main/demo.ipynb)
[![Google Drive Data](https://img.shields.io/badge/Google%20Drive-Datasets%20%26%20Outputs-blue?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1nLETsjaAwh3EDJRI9tUwJmkPd7hoM1ZD?usp=sharing)

> **📦 Evaluator Data & Proof Access:**  
> The complete multi-million record test prediction outputs (`matching_results.tsv` and `candidate_pairs.tsv` covering all 1.73M entities) and full datasets are hosted on Google Drive:  
> 🔗 **[Access Google Drive Data Folder](https://drive.google.com/drive/folders/1nLETsjaAwh3EDJRI9tUwJmkPd7hoM1ZD?usp=sharing)**

---

## 1. Problem Overview

In large-scale enterprise data ecosystems, commercial platforms, and knowledge bases, business identity information is continuously ingested from multiple independent, uncurated data sources. Each contributing stream provides partial, incomplete, and noisy fragments of information about real-world commercial establishments. Crucially, these disparate feeds share **no universal global keys or common identifiers** (such as LEI, VAT, or registration tax codes) due to regulatory constraints, disparate ingestion pipelines, and fragmented jurisdictional tracking.

The challenge of determining which records across independent sources refer to the exact same real-world business entity is known as **Entity Resolution (ER)** or **Enterprise Record Linkage**.

This project establishes a scalable, production-grade machine learning system that, given business records from 3 independent data sources with noisy and inconsistent fields, accurately resolves records across sources to identify and cluster matching commercial entities.

---

## 2. Source Data Structure & Relationship

The data landscape consists of 3 distinct data sources:

* **Source 1 ($\mathcal{S}_1$):** A deduplicated reference dataset serving as the canonical baseline.
* **Source 2 ($\mathcal{S}_2$) & Source 3 ($\mathcal{S}_3$):** Uncurated secondary sources containing noisy, duplicate, or partial records.

**Objective:** For every reference entity in Source 1, identify all corresponding matching records across Source 2 and Source 3. A Source 1 entity may match zero (singleton), one, or multiple records across the auxiliary sources.

### Data Attributes:
All records across sources contain four core semi-structured fields:

1. `entity_id`: Unique identifier for the record. The prefix indicates the originating source (`S1-`, `S2-`, or `S3-`).
2. `business_name`: The registered or commercial business trade name (subject to extreme variations, abbreviations, legal suffixes, typos, and transliterations).
3. `business_address`: Physical address string (subject to incomplete postal details, landmark-based descriptions, reordered tokens, and missing postal PINs/states).
4. `country`: Geopolitical jurisdiction label (spanning domestic markets such as the `US` and `India`, as well as open-set international jurisdictions such as `France`).

---

## 3. Real-World Noise Patterns & Analytical Challenges

In enterprise operations, reconciling commercial entity records must overcome five critical domain-specific challenges:

1. **Extreme Lexical & Corporate Suffix Inconsistencies:**
   * Heavy regional abbreviation variations (e.g., *Corp* vs. *Corporation*, *Pvt Ltd* vs. *Private Limited*, *LLC* vs. *L.L.C.*, *SAS* vs. *SARL* vs. *SASU*).
   * Doing-Business-As (DBA) trade names, colloquial brand names, and phonetic spelling corruptions.

2. **Complex, Unstructured Address Topography:**
   * Missing administrative primitives (missing PIN/postal codes, provinces, or district names).
   * Subjective navigation landmarks (particularly prevalent in emerging markets, e.g., *"Near SBI ATM, 2nd Cross, Industrial Area"*).
   * Token-order transpositions between municipal building numbers, suite numbers, and city/state names.

3. **Multilingual Diacritic Inconsistencies:**
   * International records (e.g., European/French markets) introduce variable accented glyphs (*é, è, ê, ô, ç, à*), which are inconsistently stripped, converted, or corrupted across legacy ingestion feeds.

4. **The Quadratic Comparison Bottleneck ($\mathcal{O}(NM)$):**
   * Comparing a reference pool of size $N \approx 1.73 \times 10^6$ against target candidate pools of total size $M \approx 9.97 \times 10^6$ creates a quadratic search space of $\approx 1.72 \times 10^{13}$ pairwise comparisons. Brute-force pairwise evaluation is computationally impossible under real-world SLA constraints.

5. **Asymmetric Precision-Heavy Evaluation Dynamics (Macro $F_{0.5}$):**
   * In enterprise fraud detection, AML/KYC compliance, and corporate risk management, the cost of a **false positive** (erroneously merging two distinct companies) is significantly more damaging than a missed match.
   * Furthermore, singleton entities (reference businesses with zero matches) are severely penalized: predicting any false match drops the entity score from $1.0$ directly to $0.0$.

---

## 4. Evaluation Metric: Macro-Averaged $F_{0.5}$

Model performance is evaluated using the macro-averaged $F_{0.5}$ score across all reference records $e_i \in \mathcal{S}_1$. 

For each individual reference record $e_i$, with ground truth match set $Y_i$ and predicted match set $\hat{Y}_i$:
$$\text{Precision } (P_i) = \frac{|Y_i \cap \hat{Y}_i|}{|\hat{Y}_i|}, \quad \text{Recall } (R_i) = \frac{|Y_i \cap \hat{Y}_i|}{|Y_i|}$$

$$F_{0.5}(i) = \frac{(1 + 0.5^2) \cdot P_i \cdot R_i}{0.5^2 \cdot P_i + R_i} = \frac{1.25 \cdot P_i \cdot R_i}{0.25 \cdot P_i + R_i}$$

### Boundary Conditions for Singletons:
* **True Singleton:** If $|Y_i| = 0$ and $|\hat{Y}_i| = 0 \implies F_{0.5}(i) = 1.0$
* **False Merge on Singleton:** If $|Y_i| = 0$ and $|\hat{Y}_i| \ge 1 \implies F_{0.5}(i) = 0.0$
* **Missed Match:** If $|Y_i| \ge 1$ and $|\hat{Y}_i| = 0 \implies F_{0.5}(i) = 0.0$
* **Zero Intersection:** If $|Y_i \cap \hat{Y}_i| = 0$ and $|\hat{Y}_i| \ge 1 \implies F_{0.5}(i) = 0.0$

The global objective is the macro-average across all $N$ entities:
$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^{N} F_{0.5}(i)$$

---

## 5. System Output Specifications

The enterprise resolution engine generates two standardized tab-separated (`.tsv`) operational artifacts:

1. **`output/matching_results.tsv`** (Final Entity Matches):
   Contains the verified cross-source entity matches for every reference entity.
   * Columns: `source1_entity_id`, `matched_entity_ids` (comma-separated list of matched IDs from Sources 2 and 3, or empty if singleton).
   * Exact one-to-one row mapping: Every reference entity must be represented with exactly one row.

2. **`output/candidate_pairs.tsv`** (Candidate Generation / Blocking Pool):
   Contains the high-recall candidate set produced by the sublinear blocking stage before final classification.
   * Columns: `source1_entity_id`, `candidate_entity_ids` (comma-separated candidate list).
   * Used for auditing candidate recall preservation and reduction ratios.

---

## 6. Pipeline Architecture Summary

To solve this problem at multi-million scale within strict resource constraints, our system implements a two-stage, streaming architecture:

1. **Preprocessing & Normalization:** High-speed compiled regex canonicalization (92,000 rec/s) for Unicode NFKD decomposition, case folding, and legal suffix standardization.
2. **Geographical Invariant Partitioning:** Enforces the Zero Cross-Country Invariant ($0.0000\%$ cross-border leakage across 69,144 true pairs), mathematically decomposing the global problem into disjoint country sub-spaces.
3. **Sublinear BM25-IDF Inverted Index Blocking:** Multi-field token indexing (Name: 3.0, Number: 2.5, Address: 1.5) with posting list frequency truncation and dynamic top-IDF query pruning. Achieves **97.49% candidate recall** while pruning **99.998%** of the comparison space.
4. **11-Dimensional Vectorized Feature Extraction:** RapidFuzz C++ pairwise similarity computing orthographic, token permutation, length discrepancy, and numerical sequence alignment signals at $>100,000$ pairs/second.
5. **Macro-$F_{0.5}$ Calibrated LightGBM Classification:** Gradient boosted decision trees calibrated via empirical threshold grid search ($\tau^* = 0.74$), delivering **98.42% precision, 91.24% recall, and 96.79% Macro $F_{0.5}$** across 1.73M entities in under 12 minutes on commodity CPU hardware.
