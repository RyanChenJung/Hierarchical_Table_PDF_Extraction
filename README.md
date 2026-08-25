# Visual Document Understanding: Hierarchical Table Extraction & Business Insights

An end-to-end pipeline and benchmark for extracting structured data from complex financial tables in scanned documents — from raw table images to business-ready records. Built for the Advanced Computer Vision for Deep Learning final project.

**Team:** Lawrence Lin, Ryan Chen, Kacey Zhu, Jared Maksoud

---

## Overview

Financial documents contain tables with multi-level headers, merged cells, irregular layouts, and inconsistent scan quality. Traditional OCR extracts raw text but discards the logical table structure, making downstream analytics impossible. This project builds and benchmarks a **Table Structure Recognition (TSR) → HTML → Key Information Extraction (KIE) → Structured Records** pipeline across six model architectures, evaluated on the FinTabNet dataset, and ships a deployment-ready dashboard (**TableSight**) for interactive inference and evaluation.

```
Document Image → TSR (HTML Structure) → HTML → DataFrame → KIE → Structured Business Records
```

## Dataset

**FinTabNet** — 32,670 financial table images, split into:
- Train: 850 · Validation: 150 · Test: 300 (stratified sample from hard examples, 260 images × 5 table types: wide, tall, low-contrast, normal, low-quality)

**Key EDA findings:**
- Wide tables dominate the dataset (48.6%), but **tall tables score hardest** (avg. difficulty 29) — image *type*, not image *quality*, is the primary driver of difficulty.
- Structural complexity is the core challenge: 98.1% of tables have `colspan > 1` (grouped headers), and row count correlates strongly with difficulty (r = 0.87). Image quality is nearly irrelevant (r ≈ 0).

**Preprocessing:** resolution/format standardization, HTML annotation cleanup, corrupted-sample filtering, image–HTML alignment validation. **Augmentation (training only):** small-angle rotation, scaling, contrast adjustment, blur simulation.

## Evaluation Metrics

- **TEDS (Tree Edit Distance Similarity):** 0–1 score (1.0 = perfect match), evaluated in two modes — **Skeleton TEDS** (structure only) and **Pipeline TEDS** (full output, structure + content).
- **Entity-level F1** for KIE — correctness and completeness of extracted structured information.

## Models Benchmarked

| Model | Type | Approach |
|---|---|---|
| **UniTable** | Trainable | Pixel-to-text encoder-decoder (ConvStem + Transformer), 3-pass pipeline (structure → bbox → content), LoRA fine-tuned |
| **SPARTAN** | Heuristic CV | No trainable weights — adaptive thresholding → OCR clustering → grid reconstruction, CPU-only |
| **Florence-2** | Trainable | Unified vision-language model, prompt-based generation, LoRA fine-tuned for TSR/KIE |
| **ChatGPT** | External API | GPT-4o-mini multimodal, prompt-based reasoning |
| **LlamaParse** | External API | Production document parsing API (OCR + layout analysis) |

## Results

| Model | Skeleton TEDS | Pipeline TEDS | F1 Score |
|---|---|---|---|
| UniTable (Fine-tuned) | **0.8353** ⭐ | 0.3417 | 0.4928 |
| UniTable (Original) | 0.8353 | 0.3389 | 0.4829 |
| **SPARTAN** | 0.7253 | **0.5391** ⭐ | **0.70** ⭐ |
| Florence-2 (LoRA) | 0.0441 | 0.0287 | N/A |
| ChatGPT | N/A | 0.5109 | N/A |
| LlamaParse | N/A | 0.5120 | N/A |

⭐ = best in category. SPARTAN leads on low-contrast images; ChatGPT collapses on tall tables; LlamaParse is the most consistent across image types.

### Key Findings

1. **Structure ≠ Content.** The best structure model (UniTable, 0.8353 Skeleton TEDS) is not the best content model (SPARTAN, 0.5391 Pipeline TEDS) — the two should be optimized separately.
2. **Cascading errors dominate.** Multi-stage pipelines amplify small mistakes: a single mispredicted structural token in UniTable shifts every downstream cell, while SPARTAN's heuristic, geometry-tied approach confines errors locally — beating UniTable on Pipeline TEDS despite weaker raw structure recognition.
3. **Florence-2 fine-tuning underperformed its own zero-shot baseline** (0.0287 vs. 0.1539 Pipeline TEDS), traced to a large domain gap (pre-trained for OCR/captioning/grounding, not HTML generation), training instability (early-stopped at epoch 17/50), and likely LoRA misconfiguration.
4. **SPARTAN's main weakness is span loss** — 54.2% of its failures involve lost column spans in multi-level headers.

## TableSight: Deployment Dashboard

An interactive dashboard for running and evaluating the pipeline, with four tabs:

- **Compare** — single-image inference (upload or sample from a 1,300-image gallery); runs all models in parallel with side-by-side TEDS scores.
- **Benchmark** — pre-computed aggregate stats over 1,300 samples, broken down by image type and difficulty.
- **Batch Eval** — run all models over an uploaded CSV slice and export per-sample scores.
- **Failure Analysis** — surfaces all samples with TEDS < 0.5 alongside a diff against ground truth.

Also includes an **HTML → DataFrame export** utility (colspan expansion, rowspan carry-down, short-row padding) — cleanly parses 1,495 of 1,510 complex samples.

## Future Work

- **Fine-tune UniTable further** — more epochs, higher learning rate, increased LoRA impact.
- **Learned row-grouping classifier** — replace SPARTAN's fixed `row_tol_frac` threshold with a learned classifier to directly target span loss (54.2% of failures) without disrupting multi-line wrapped layouts.
- **Hybrid architecture (recommended):** UniTable for structure + SPARTAN for OCR/content + a parsing layer — composing each model's strengths rather than relying on one model for the full pipeline.

## Team

- Lawrence Lin
- Ryan Chen
- Kacey Zhu
- Jared Maksoud

📊 [Slides](https://github.com/RyanChenJung/Hierarchical_Table_PDF_Extraction/blob/main/Hierarchical_Table_PDF_Extraction%20Presentation.pdf) · 📄 [Report](https://github.com/RyanChenJung/Hierarchical_Table_PDF_Extraction/blob/main/Final%20Report.pdf)

*Advanced Computer Vision for Deep Learning — Final Project*
