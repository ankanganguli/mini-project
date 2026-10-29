# 00_MASTER_RULES.md

## Core Invariants

- **Invariant 1 (Zero Mock Policy):** Strictly zero dummy scores, hardcoded percentages, or `random.uniform()` calls. All outputs must come from active scikit-learn models and OpenCV visual forensics.
- **Invariant 2 (Dynamic Dataset Ingestion
 The ML engine must accept arbitrary CSV datasets, automatically identify text feature columns and binary target labels, and handle imbalanced distributions.

- **Invariant 3(Single-Task Execution):** Work strictly against `.brain/15_MICROTASKS.md`. Never build ahead or tackle multiple tasks in a single turn.
