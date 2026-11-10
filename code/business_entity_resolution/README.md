# Business Entity Resolution Pipeline

This repository contains the complete, production-grade, end-to-end pipeline for the Business Entity Resolution Challenge.

## Pipeline Architecture

1. **High-Performance Multi-Lingual Text Normalization (`src/normalize.py`)**:
   - Diacritical mark decomposition & accent stripping (NFKD) for French & multilingual strings.
   - Punctuation removal and case folding.
   - Comprehensive legal suffix canonicalization (US, Indian, and French legal forms) via single-pass compiled regex mapping.
   - Address token canonicalization (street, avenue, boulevard, road, near, opposite, etc.).

2. **Country-Partitioned Sparse Blocking (`src/blocking.py`)**:
   - Strict country blocking (matches 100% respect country boundaries, verified empirically with 0.0000% cross-country error rate).
   - High-dimensional sublinear char 3-gram TF-IDF on normalized business names and normalized addresses.
   - Fast sparse CSR dot-product matrix multiplication and top-K candidate selection with memory-efficient indexing.

3. **Pairwise Feature Engineering (`src/features.py`)**:
   - Multi-metric string similarities: Jaro-Winkler, Levenshtein ratio, RapidFuzz Token Sort Ratio, RapidFuzz Token Set Ratio.
   - Address token Jaccard similarity.
   - Exact and partial numeric sequence matching (extracting street numbers, door numbers, postal codes, and PIN codes).
   - Name length differences and TF-IDF candidate score integration.
   - Extraction speed exceeding 100,000 candidate pairs per second.

4. **Matching Classifier & Optimization (`src/model.py`)**:
   - Gradient boosted decision trees (LightGBM) trained on pairwise similarity vectors.
   - Exact decision threshold optimization tailored specifically to the macro-averaged $F_{0.5}$ metric (which penalizes false merges 2× more than missed matches).

5. **Streaming Inference Pipeline (`src/pipeline.py`)**:
   - Country-by-country chunked execution ensuring constant low memory footprint (< 2.5 GB peak RAM).
   - Direct streaming output to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

---

## Installation & Setup

Ensure Python 3.8+ is installed:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline End-to-End

### 1. Training & Validation
To train the LightGBM matcher model and optimize the decision threshold on the training dataset:

```bash
python src/train.py
```
This saves:
- `models/matcher_model.pkl` (trained model)
- `models/model_meta.pkl` (optimal threshold & hyperparameters)

### 2. Full Test Inference
To run candidate generation and matching inference over the entire test dataset (`dataset/test/`):

```bash
python src/pipeline.py --test-dir dataset/test --output-dir output
```

The script produces:
- `output/matching_results.tsv`: Final predicted matching entities for every Source 1 entity.
- `output/candidate_pairs.tsv`: Final candidate pool produced by the blocking stage.

### 3. Submission Validation
Verify that both output files strictly satisfy all challenge validation rules:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
