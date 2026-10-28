# MTUOC-eval

**MTUOC-eval** is a standalone desktop application for Machine Translation (MT) evaluation and Quality Estimation (QE) developed with **PyQt6**. It provides an intuitive graphical interface for benchmarking machine translation models and systems using standard rule-based metrics and modern neural models with local GPU acceleration.

Developed by the **MTUOC**project.

---

## Features

- **Desktop Interface (PyQt6)**: Fast, responsive native GUI featuring dynamic system management and asynchronous evaluation workers.
- **Dual Evaluation Modes**:
  - **Reference-Based Evaluation**: Assess MT hypotheses against human gold references.
  - **Referenceless Quality Estimation (QE)**: Evaluate translation quality using source texts when references are unavailable.
- **Supported Metrics**:
  - **Rule-based & Edit-distance**: BLEU (SacreBLEU), chrF++, TER (Translation Edit Rate), Character Edit Distance (%EdDist), and Word Error Rate (WER).
  - **Neural & Embedding-based**: COMET, COMET-QE, BERTScore, and SBERT (Cosine Similarity).
- **GPU & Hardware Acceleration**:
  - Optimized for CUDA-enabled GPUs (including modern RTX architectures).
  - Automatic fallback to CPU if GPU resources or memory are constrained.
- **Segment-Level Analysis**:
  - Detailed segment view with side-by-side comparison (Source, Reference, Hypothesis).
  - Inline difference highlighting (`DIFF`) showing insertions and deletions.
- **Visual Analytics**:
  - Bar charts comparing system scores across all evaluated metrics.
  - Radar / Spider charts offering system performance footprints and comparisons.
  - High-resolution PNG export for all generated charts.
- **Export & Reporting**:
  - Export corpus-level summaries to publication-ready **LaTeX (`.tex`)** tables.
  - Comprehensive reporting in **Excel (`.xlsx`)** and **CSV**.
- **Reproducibility**:
  - Automatic generation and logging of reproducibility signatures for all metrics (SacreBLEU, COMET, BERTScore).
