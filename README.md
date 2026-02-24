# Surgical Gesture Consensus Framework

Code repository supporting the development of a standardized surgical gesture taxonomy through LLM-powered clustering and multi-phase community consensus validation. This project was conducted as part of the 2026 SAGES NBT Innovation Weekend by the AI steering committee.

## Overview

This project develops a standardized framework for surgical gestures that minimizes definitional overlap and is generalizable across surgical procedures. The framework supports multiple applications including video-based assessment (VBA), computer vision (CV), and autonomous surgical system development.

### Research Objectives

- Minimize overlap between surgical gesture definitions
- Create a taxonomy generalizable across procedures and approaches (open, laparoscopic, robotic)
- Support multiple use cases: performance evaluation, computer vision models, and autonomous surgery
- Establish a common language for surgical research and clinical innovation
- Improve scientific rigor by reducing annotation variability and enabling reproducible cross-study comparisons

## Research Methodology

This repository documents a multi-phase consensus process conducted with 75+ surgical experts:

1. **Definition Collection & Curation** - Aggregated 270 gesture terms from 75 peer-reviewed studies; filled missing definitions using LLM prompting
2. **Definition Normalization** - Standardized definitions into clear, narrative descriptions
3. **Phase 1 Clustering** - LLM-powered agglomerative clustering: 270 gestures → 106 gestures in 11 clusters
4. **Community Feedback (Round 1)** - Online survey with 45 expert responders; consensus-driven refinement
5. **Phase 2 Clustering** - Re-clustered refined list: 107 gestures → 97 gestures in 34 clusters
6. **Hierarchical Taxonomy** - Introduced 3-level hierarchy (Clusters → Gestures → Sub-Gestures) based on use-case requirements
7. **Community Feedback (Round 2)** - 36 participants validated cluster appropriateness via agree/disagree survey
8. **Consensus Refinement** - Expert team adjustments: 97 gestures → 10 clusters, 25 gestures, 45 sub-gestures
9. **Video Annotation Validation** - Interactive UI with 35 participants annotating 30 surgical video clips
10. **Live In-Person Poll** - 2026 SAGES NBT Innovation Weekend: 8 specific consensus questions with 80%+ voting threshold
11. **Final Gesture List** - Published consensus taxonomy with minor refinements based on live poll results

### Technical Features

- **LLM-Powered Embedding** - `sentence-transformers` for semantic gesture analysis
- **Agglomerative Clustering** - Hierarchical clustering with dendrogram visualization
- **Quality Metrics** - Silhouette scoring for cluster homogeneity assessment
- **Data Processing** - Robust CSV handling with multi-encoding support
- **Community Validation** - Integration of expert feedback throughout pipeline

## Project Structure

```
LLMClustering/
├── src/                           # Main source code
│   ├── clustering/
│   │   └── cluster_analyzer.py   # Clustering algorithms and evaluation
│   ├── tokenizer/
│   │   ├── llm_tokenizer.py      # Sentence-transformer embeddings
│   │   └── simple_tokenizer.py   # TF-IDF vectorization
│   └── utils/
│       ├── prepare_data.py       # Data loading and processing
│       └── file_utils.py         # File I/O utilities
├── notebooks/                     # Analysis and clustering pipelines
│   ├── Agglomerative Clustering (Phase 01).ipynb      # First round clustering (270 → 106 gestures)
│   ├── Agglomerative Clustering (Phase 02).ipynb      # Second round clustering (107 → 97 gestures)
│   └── Normalization of Definitions.ipynb             # Definition standardization pipeline
├── data/                          # Data sources
│   └── example_list.csv          # Gesture definitions compiled from 75 studies
└── requirements.txt               # Python dependencies
```

## Installation

1. **Clone the repository** (if applicable):
```bash
git clone <repository-url>
cd LLMClustering
```

2. **Create and activate a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Dependencies

- **transformers** (≥4.30.0) - For LLM model access
- **torch** (≥2.0.0) - Deep learning framework
- **scikit-learn** (≥1.3.0) - Clustering and evaluation metrics
- **numpy** (≥1.24.0) - Numerical computing
- **pandas** (≥2.0.0) - Data manipulation
- **matplotlib** (≥3.7.0) - Visualization
- **seaborn** (≥0.12.0) - Statistical visualization
- **tqdm** (≥4.65.0) - Progress bars
- **python-dotenv** (≥1.0.0) - Environment variable management

## Quick Start

### Using the Clustering Pipeline

To reproduce the consensus clustering analysis:

1. **Prepare gesture data** from [data/example_list.csv](data/example_list.csv):
```python
from src.utils.prepare_data import read_gestures
from pathlib import Path

gestures = read_gestures(Path("data/example_list.csv"))
print(f"Loaded {len(gestures)} gestures")
```

2. **Generate embeddings** using sentence-transformers:
```python
from src.tokenizer.llm_tokenizer import LLMTokenizer

tokenizer = LLMTokenizer(model_name="sentence-transformers/all-mpnet-base-v2")
gesture_texts = [g['content'] for g in gestures]
embeddings = tokenizer.encode(gesture_texts, batch_size=32)
```

3. **Perform clustering and evaluation**:
```python
from src.clustering.cluster_analyzer import ClusterAnalyzer

analyzer = ClusterAnalyzer()
labels = analyzer.kmeans_clustering(embeddings, n_clusters=10)
score = analyzer.evaluate_clustering(embeddings)
print(f"Silhouette Score: {score:.3f}")
```

### Full Workflow

For the complete agglomerative clustering pipeline with dendrogram visualization and cluster naming, see the notebooks:
- [Agglomerative Clustering (Phase 01).ipynb](notebooks/Agglomerative%20Clustering%20%28Phase%2001%29.ipynb) - Initial clustering from 270 gestures
- [Agglomerative Clustering (Phase 02).ipynb](notebooks/Agglomerative%20Clustering%20%28Phase%2002%29.ipynb) - Refined clustering with consensus feedback

## API Reference

### Data Utilities

#### `prepare_data.read_gestures(csv_path)`
Reads gesture definitions from CSV file with robust encoding handling.

**Parameters:**
- `csv_path` (Path): Path to CSV with columns `gesture_name` and `gesture_definition`

**Returns:**
- List[Dict]: List of gesture objects with `id` and `content` fields

---

### Tokenization

#### `LLMTokenizer(model_name="sentence-transformers/all-mpnet-base-v2")`
Generates semantic embeddings using sentence-transformers.

**Methods:**
- `encode(texts, batch_size=32)` → np.ndarray: Returns embedding matrix (n_texts, embedding_dim)

#### `SimpleTokenizer()`
Lightweight TF-IDF vectorization alternative.

**Methods:**
- `encode(texts, batch_size=32)` → np.ndarray: Returns sparse-to-dense embedding matrix

---

### Clustering & Evaluation

#### `ClusterAnalyzer()`
Performs clustering and quality assessment on embeddings.

**Methods:**
- `kmeans_clustering(embeddings, n_clusters=5)` → np.ndarray: K-means cluster labels
- `dbscan_clustering(embeddings, eps=0.5, min_samples=5)` → np.ndarray: DBSCAN cluster labels
- `evaluate_clustering(embeddings)` → float: Silhouette score (range: -1 to 1, higher is better)

## Notebooks

### Data Preparation
**Normalization of Definitions.ipynb**
- Standardizes gesture definitions from heterogeneous sources
- LLM-powered narrative rephrasing for consistency
- Handles missing definitions by referencing source literature

### Phase 1: Initial Clustering & Curation
**Agglomerative Clustering (Phase 01).ipynb**
- Generates embeddings from gesture definitions
- Deduplicates equivalent gestures
- Performs hierarchical agglomerative clustering
- Names clusters using LLM-generated labels and definitions
- Evaluates cluster homogeneity with silhouette scoring

### Phase 2: Consensus-Driven Refinement
**Agglomerative Clustering (Phase 02).ipynb**
- Clusters refined gestures (after incorporating expert feedback responses)

---

**Run notebooks with:**
```bash
jupyter notebook
```


## Configuration & Parameters

For reproducing the clustering analysis, key parameters used:

```python
# Embedding model
model_name = "sentence-transformers/all-mpnet-base-v2"
batch_size = 32

# Agglomerative clustering
linkage = "ward"  # or other linkage methods for dendrograms
```

## Troubleshooting

**CUDA Memory Errors with LLM Tokenizer**
- Reduce `batch_size` in `llm_tokenizer.encode()` method
- Alternatively, use CPU-only PyTorch installation

**CSV Encoding Issues**
- The `prepare_data.py` script automatically attempts UTF-8, UTF-8-sig, and Latin-1 encodings
- Handle BOM and special characters automatically

**Missing Pretrained Models**
- Sentence-transformer models download automatically on first use
- Requires active internet connection

**Reproducibility**
- All random states set to 42 for algorithm reproducibility
- LLM outputs may vary slightly due to model updates

## Contributing

This repository documents the consensus process as of February 2026. Future development will focus on:
- Temporal boundary definitions for surgical gestures
- Support for hierarchical action triplets/quadruplets
- OR technology standardization

Contributions should maintain the consensus-based framework and include expert validation studies.

## License

This research was conducted as part of the 2026 SAGES NBT Innovation Weekend proceedings. Please cite appropriately if using this taxonomy in your research.

## Community Contributors

This consensus framework was developed through collaboration with:
- **Surgical experts, Engineers, and Industry Partners** from academic and clinical institutions
- **Survey Respondents** providing feedback on clustering approach and evaluating cluster appropriateness  
- **Video Annotation Participants** validating gesture recognition via surgical video clips
- **2026 SAGES NBT Innovation Weekend attendees** refining final taxonomy

## Project Lead

Maria Clara Morais, MD
Postdoctoral Research Fellow – Intraoperative Performance Analytics Lab
Northwell Health, New York, NY

Filippo FIlicori, MD
Associate Professor of Surgery
Chair, SAGES AI Committee
System Chief Surgical Innovation
Program Director, Minimally Invasive Surgery
Northwell Health, New York


---

**Event:** 2026 SAGES NBT Innovation Weekend
**Date:** February 19, 2026  
**Final Update:** February 2026
