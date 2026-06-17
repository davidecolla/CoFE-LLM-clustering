# Beyond Context Windows: Evaluating Structured LLM-Based Clustering of Large-Scale Multilingual Corpora

This repository contains the code accompanying the paper:

> **Beyond Context Windows: Evaluating Structured LLM-Based Clustering of Large-Scale Multilingual Corpora**  
> Davide Colla, Elisa Di Nuovo, Nicolas Stefanovitch, Bertrand De Longueville, Charles MacMillan  
> European Commission, Joint Research Centre

## Overview

We propose a structured LLM-based clustering pipeline that overcomes context-window limitations of Large Language Models through **chunking**, **local clustering**, **cluster merging**, and **zero-shot classification**. The approach is evaluated on the Conference on the Future of Europe (CoFE) dataset — a real-world multilingual corpus of ~8,000 citizen-generated proposals in 24+ EU languages.

The pipeline identifies both **macro-categories** (10 coarse-grained CoFE themes) and **micro-categories** (fine-grained topics within each theme), using LLaMA 3.3 70B for all LLM-based operations.

As a diagnostic baseline, we also implement an **autonomous LLM-based coding agent** (LangGraph-based) that, given the same task, converges on a classical TF-IDF + k-means pipeline — highlighting that methodological structure outweighs agentic autonomy for this task.

## Repository Structure

```
.
├── paper.tex                   # Research paper (ACL format)
├── data/                       # Dataset files
│   ├── proposals_tiny.json             # 8K proposal titles (id, text, language)
│   ├── proposals_tiny_translation.json # Same with English translations (text_en)
│   ├── proposals_tiny_full.json        # Extended version of proposals
│   ├── proposals.json                  # Full proposals dataset
│   └── plan.json                       # Agent-generated execution plan
├── nokebooks/                  # Notebooks for the structured LLM pipeline
│   ├── 1_create_clusters.ipynb         # Chunking + local clustering + merging
│   ├── 2_classify_texts.ipynb          # Zero-shot text classification
│   ├── 3_embed_proposals.ipynb         # Embedding with multiple models
│   ├── 4_analysis.ipynb                # Quantitative evaluation (silhouette, V-measure, F1)
│   ├── 5_sampling.ipynb                # Sampling for human/LLM evaluation
│   ├── 6.second_level_clustering.ipynb # Micro-category clustering
│   ├── 7.non_overlapping_clustering.ipynb # Non-overlapping experiments
│   ├── 8.llm-judge.ipynb              # LLM-as-a-judge evaluation
│   └── eval_Llama-3_babilong.ipynb    # BABILong benchmark evaluation
├── agent/                      # Agentic LLM-based coding baseline
│   ├── app.py                          # Entry point: builds and runs the agent
│   ├── src/
│   │   ├── graph.py                    # LangGraph state graph (planner → writer → runner → fixer)
│   │   ├── agents.py                   # Agent definitions (planner, code writer, fixer, assistant)
│   │   └── utils.py                    # Utility functions
│   └── run/                            # Generated pipeline scripts (agent output)
│       ├── main.py                     # Combined clustering script
│       └── *.py                        # Step-by-step generated functions
└── README.md
```

## Methodology

### Structured LLM-Based Pipeline (main approach)

1. **Chunking** — Partition 8K proposals into manageable chunks (tested at 16k and 20k tokens), with optional 50% overlap to mitigate the independence assumption across chunk boundaries.
2. **Local Clustering** — Each chunk is processed independently by LLaMA 3.3 70B, which identifies cluster names and descriptions capturing semantic structure.
3. **Cluster Merging** — The LLM consolidates local cluster definitions into a unified set of global clusters (10 for macro-categories).
4. **Text Classification** — Individual proposals are assigned to global clusters via zero-shot classification based on cluster descriptions.
5. **Filtering (optional)** — Embedding-based cosine similarity filtering removes low-confidence assignments using centroid similarity thresholds.

### Agentic Baseline

An autonomous LangGraph-based agent that:
- Uses LLaMA 3.3 70B for planning and Qwen Coder 2.5 for code generation
- Autonomously decomposes the clustering task, generates Python code, executes it, and iteratively fixes errors
- Consistently converges on a classical TF-IDF + k-means pipeline

## Data

The dataset consists of ~8,000 proposal titles from the Conference on the Future of Europe, in JSON format:

```json
{
    "text": "Proposal title text",
    "id": "pr-000000309",
    "language": "fr"
}
```

The translation file adds an `"text_en"` field with English translations for non-English proposals.

## Embedding Models

The following multilingual embedding models are used for evaluation and filtering:
- **E5** — `intfloat/multilingual-e5-large`
- **Qwen** — `Qwen/Qwen3-Embedding-8B` (also tested with Matryoshka reduction to 128 dimensions)
- **BERT** — `google-bert/bert-base-multilingual-cased`
- **LASER** — Facebook's Language-Agnostic SEntence Representations

## Key Results

| Metric | Best Score | Condition |
|--------|-----------|-----------|
| V-measure (macro, unfiltered) | 0.383 | 20k chunk, overlap |
| V-measure (macro, filtered) | 0.532 | 16k chunk, overlap, BERT |
| Silhouette (agent baseline) | 0.016 | TF-IDF + k-means |
| V-measure (micro) | 0.714 | LLM-as-judge ground truth |
| Cohen's κ (human agreement) | 0.785 | Binarised Likert scale |

## Requirements

### Structured Pipeline (Notebooks)

- Python 3.10+
- pandas, numpy, scikit-learn
- sentence-transformers / transformers (for embedding models)
- torch
- LASER (for LASER embeddings)
- Access to an OpenAI-compatible LLM endpoint (LLaMA 3.3 70B)

### Agentic Baseline

- langchain, langgraph
- langchain-openai
- pydantic
- scikit-learn, numpy
- tqdm
- Access to an OpenAI-compatible LLM endpoint (LLaMA 3.3 70B, Qwen Coder 2.5)

## Usage

### Running the Structured Pipeline

The structured pipeline is implemented across the numbered notebooks in `nokebooks/`. Execute them in order:

1. `1_create_clusters.ipynb` — Performs chunking, local clustering, and merging
2. `2_classify_texts.ipynb` — Classifies all proposals into the identified clusters
3. `3_embed_proposals.ipynb` — Computes embeddings for filtering and evaluation
4. `4_analysis.ipynb` — Computes silhouette, V-measure, and F1 scores
5. `5_sampling.ipynb` — Samples proposals for qualitative evaluation
6. `6.second_level_clustering.ipynb` — Runs micro-category clustering
7. `7.non_overlapping_clustering.ipynb` — Ablation: non-overlapping experiments
8. `8.llm-judge.ipynb` — LLM-as-a-judge evaluation on sampled proposals

### Running the Agentic Baseline

```bash
cd agent
python app.py
```

The agent will plan, generate, execute, and iteratively fix the clustering pipeline. Output files (`output_clusters.json`, `results.txt`) are saved in the `agent/` directory.

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{colla2025beyond,
  title={Beyond Context Windows: Evaluating Structured LLM-Based Clustering of Large-Scale Multilingual Corpora},
  author={Colla, Davide and Di Nuovo, Elisa and Stefanovitch, Nicolas and De Longueville, Bertrand and MacMillan, Charles},
  year={2025}
}
```

## License

This repository is provided for research purposes. Please refer to the paper for full details on methodology and evaluation.
