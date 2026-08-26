# Multilingual Language Identification

A reproducible undergraduate machine-learning project for identifying language from two independent modalities: multilingual text and pre-extracted speech features. This revision updates the **Text Language Identification** track to reproduce the feature engineering, model families, tuning grids, and results of the supplied reference project while retaining the modular `src/` application architecture.

The required baseline is local, CPU-compatible, Poetry-managed, and does not require a paid or cloud API.

## Project highlights

- independent Speech and Text datasets;
- group-aware train/test splitting to reduce source leakage;
- five-fold group-aware cross-validation for Text;
- complete hyperparameter tuning on the training partition;
- four Text classification families;
- confusion matrix and complete classification report for every classifier;
- Accuracy, Precision, Recall, Macro F1, and weighted F1 evaluation;
- label-free Text clustering with K-Means, DBSCAN, Agglomerative Clustering, and Gaussian Mixture;
- fitted models, predictions, metrics, plots, and clustering assignments saved as artifacts.

## System architecture

```text
Raw text files
    |
    v
Data discovery and UTF-8 validation
    |
    v
Whitespace normalization + source-group inference
    |
    +---------------- Supervised classification ----------------+
    |                                                            |
    |  Group-aware holdout (397 train / 211 test)                 |
    |       -> 5-fold StratifiedGroupKFold                        |
    |       -> Character + Word TF-IDF                            |
    |       -> Chi-square top-50 feature selection                |
    |       -> hyperparameter tuning                              |
    |       -> final held-out evaluation and model bundle         |
    |                                                            |
    +---------------- Unsupervised clustering --------------------+
            Character + Word TF-IDF fitted without labels
            -> TruncatedSVD (50 components)
            -> L2 normalization
            -> K-Means / DBSCAN / Agglomerative / Gaussian Mixture
            -> internal metrics
            -> post-fit ARI, NMI, purity, and composition
```

Reusable implementation lives in `src/language_identification/`. Files in `scripts/` are thin command-line entry points, `configs/` contains reproducible settings, `tests/` contains automated checks, and generated files are written below `artifacts/`.

## Repository structure

```text
Language_Identification/
├── configs/
│   ├── base.yaml
│   ├── speech.yaml
│   └── text.yaml
├── data/raw/text/                         # immutable Text input
├── Speech_Identification/                 # supplied Speech inputs
├── src/language_identification/
│   ├── common/
│   ├── speech/
│   └── text/
├── scripts/
│   ├── run_text_preprocessing.py
│   ├── run_text_classification.py
│   ├── run_text_clustering.py
│   └── predict_text.py
├── artifacts/text/                        # generated Text results
├── tests/
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Dataset

### Text dataset

The Text dataset contains 608 independent UTF-8 documents in six languages. The directory name supplies the label; the second directory level optionally supplies gender metadata. Speech rows are not used to create or pair Text samples.

| Language | Documents |
|---|---:|
| German | 104 |
| Italian | 106 |
| Japanese | 100 |
| Korean | 98 |
| Portuguese | 100 |
| Spanish | 100 |
| **Total** | **608** |

Data-quality checks found no empty document, no exact duplicate after cleaning, and no Unicode replacement character. Text lengths range from 159 to 423 characters, with a median of 315.

### Group-aware split

A nine-digit filename prefix is treated as the source group when available. Files without this prefix receive a unique path-based group. The split is the first deterministic fold of `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)`.

| Partition | Documents | Source groups |
|---|---:|---:|
| Training | 397 | 193 |
| Held-out test | 211 | 91 |

Source-group overlap and normalized-text overlap between training and test are both zero.

## Feature engineering

Cleaning collapses consecutive whitespace and trims surrounding whitespace. It deliberately preserves scripts, case, punctuation, digits, accents, and diacritics.

The supervised Text representation is fitted inside every training or cross-validation fold:

| Feature family | Configuration |
|---|---|
| Character TF-IDF | `analyzer="char_wb"`, n-grams `(2, 4)`, `max_features=3000`, `min_df=1`, sublinear TF |
| Word TF-IDF | `analyzer="word"`, n-grams `(1, 2)`, `max_features=2000`, `min_df=1`, sublinear TF |
| Combination | `FeatureUnion` |
| Selection | `SelectKBest(chi2, k=50)` |

Clustering uses the combined TF-IDF representation without Chi-square because Chi-square requires language labels. It then applies `TruncatedSVD(n_components=50)` and L2 normalization without labels.

## Algorithms

### Classification and hyperparameter tuning

All candidates are evaluated with five-fold `StratifiedGroupKFold`, using validation Macro F1 as the selection metric. The held-out test set is excluded from tuning and family selection.

| Family | Fixed settings | Tuned parameter values | Selected value |
|---|---|---|---:|
| KNN | cosine distance, brute-force search, uniform weights | `n_neighbors`: 1, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250 | 1 |
| Decision Tree | `random_state=42` | `max_depth`: 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20 | 10 |
| Logistic Regression | balanced class weights, `max_iter=1000`, `random_state=42` | `C`: 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100 | 100 |
| Multinomial Naive Bayes | standard multinomial model | `alpha`: 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200 | 0.001 |

The complete candidate-level CV tables are saved as `artifacts/text/classification/metrics/<model>_cv_results.csv`.

### Clustering

Language labels are never passed to feature fitting, dimensionality reduction, candidate selection, or cluster fitting.

| Family | Candidate search | Selected parameters |
|---|---|---|
| K-Means | `k=2..8`, maximum Silhouette | `n_clusters=6`, `n_init=20` |
| DBSCAN | `min_samples in {5,10}` and neighbor-distance quantiles `{0.70,0.80,0.90,0.95}` | `eps=0.6375517`, `min_samples=5` |
| Agglomerative | `k=2..8`, linkage in `{ward, average}`, maximum Silhouette | `n_clusters=6`, `linkage="ward"` |
| Gaussian Mixture | `k=2..8`, covariance type diag, minimum BIC selection | selected by minimum BIC |‍‍‍


ARI, NMI, purity, and language composition are calculated only after cluster assignments exist and are used for interpretation rather than fitting.

## Installation

Requirements:

- Python 3.12;
- Poetry 2.x;
- a CPU environment; GPU is not required.

Install the locked environment from the repository root:

```powershell
poetry install
```

The core versions used for the reproduced results are:

| Package | Version |
|---|---:|
| NumPy | 2.2.6 |
| Pandas | 2.2.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.10.3 |

## Running the project

Run the Text stages in order:

```powershell
poetry run python scripts/run_text_preprocessing.py
poetry run python scripts/run_text_classification.py
poetry run python scripts/run_text_clustering.py
```

Run automated tests:

```powershell
poetry run pytest
```

Predict the language of new raw text with the selected full pipeline:

```powershell
poetry run python scripts/predict_text.py --text "Questo è un breve esempio in italiano."
```

## Results

All values below come from the executed artifact CSV and JSON files. No metric was manually edited.

### Classification results

Models are ranked by five-fold group-aware CV Macro F1.

| Rank | Model | Best parameter | CV Macro F1 | Train accuracy | Test accuracy | Test macro precision | Test macro recall | Test Macro F1 | Test weighted F1 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Logistic Regression | `C=100` | 0.9906 | 0.9924 | 0.9953 | 0.9949 | 0.9944 | 0.9946 | 0.9953 |
| 2 | KNN | `n_neighbors=1` | 0.9770 | 0.9899 | 0.9858 | 0.9851 | 0.9840 | 0.9843 | 0.9858 |
| 3 | Decision Tree | `max_depth=10` | 0.9537 | 1.0000 | 0.9858 | 0.9842 | 0.9837 | 0.9838 | 0.9858 |
| 4 | Multinomial Naive Bayes | `alpha=0.001` | 0.8813 | 0.9798 | 0.9810 | 0.9777 | 0.9792 | 0.9773 | 0.9807 |

**Selected classifier:** Logistic Regression, because it has the highest group-aware CV Macro F1. Its held-out score is reported only after selection.

#### Selected-model classification report

| Label | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| German | 1.0000 | 1.0000 | 1.0000 | 52 |
| Italian | 1.0000 | 1.0000 | 1.0000 | 26 |
| Japanese | 1.0000 | 1.0000 | 1.0000 | 33 |
| Korean | 1.0000 | 1.0000 | 1.0000 | 38 |
| Portuguese | 0.9697 | 1.0000 | 0.9846 | 32 |
| Spanish | 1.0000 | 0.9667 | 0.9831 | 30 |
| **Macro average** | **0.9949** | **0.9944** | **0.9946** | **211** |
| **Weighted average** | **0.9954** | **0.9953** | **0.9953** | **211** |

The selected classifier made one held-out error: one Spanish document was predicted as Portuguese.

### Clustering results

| Model | Clusters | Noise rate | Silhouette | Calinski-Harabasz | Davies-Bouldin | ARI | NMI | Purity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DBSCAN | 13 | 0.2352 | 0.4265 | 80.9701 | 1.0614 | 0.8089 | 0.8673 | 0.8882 |
| K-Means | 6 | 0.0000 | 0.3511 | 152.6121 | 1.4845 | 1.0000 | 1.0000 | 1.0000 |
| Agglomerative | 6 | 0.0000 | 0.3511 | 152.6121 | 1.4845 | 1.0000 | 1.0000 | 1.0000 |
| Gaussian Mixture | 8 | 0.0000 | 0.2599 | 116.5118 | 1.7507 | 0.8942 | 0.9404 | 1.0000 |

DBSCAN has the highest valid label-free Silhouette score, but it creates 13 clusters and marks 23.52% of documents as noise. K-Means and Agglomerative independently recover six clusters that align perfectly with the six supplied language labels after fitting. This confirms that the languages are highly separable in the selected text representation, while also illustrating why internal and external clustering metrics must be interpreted together.

## Generated outputs

### Classification

- `artifacts/text/classification/metrics/comparison.csv` — complete model comparison;
- `artifacts/text/classification/metrics/<model>.json` — parameters and metrics;
- `artifacts/text/classification/metrics/<model>_cv_results.csv` — all tuning candidates;
- `artifacts/text/classification/metrics/<model>_classification_report.csv` — complete per-class report;
- `artifacts/text/classification/plots/<model>_confusion_matrix.png` — confusion matrix;
- `artifacts/text/classification/predictions/<model>.csv` — held-out predictions;
- `artifacts/text/classification/models/<model>.joblib` — fitted full pipeline;
- `artifacts/text/classification/models/best_classifier.joblib` — selected inference bundle.

### Clustering

- `artifacts/text/clustering/metrics/comparison.csv` — model comparison;
- `artifacts/text/clustering/metrics/<model>.json` — parameters and metrics;
- `artifacts/text/clustering/models_or_configs/<model>_candidates.json` — candidate search;
- `artifacts/text/clustering/assignments/<model>.csv` — document assignments;
- `artifacts/text/clustering/assignments/<model>_sizes.csv` — cluster sizes;
- `artifacts/text/clustering/assignments/<model>_language_composition.csv` — post-fit composition;
- `artifacts/text/clustering/plots/<model>_clusters.png` — cluster visualization.

## Limitations

- The dataset is small and contains only six Text languages.
- The nine-digit source identifier is inferred from filenames and is not independently verified metadata.
- Unmatched files receive unique groups, so source relationships for those files cannot be inferred.
- Strong separation may partly reflect different scripts and source-specific writing patterns.
- A single held-out group fold does not quantify uncertainty as well as repeated nested evaluation.
- The clustering projection and density structure are representation-dependent.
- Speech and Text are independent datasets, so their scores are not a controlled paired comparison.

## Future work

- evaluate on external documents from unseen sources and domains;
- repeat nested group-aware cross-validation and report confidence intervals;
- add more languages, especially closely related languages using the same script;
- evaluate robustness to short, noisy, transliterated, and code-switched text;
- compare classical TF-IDF with compact pretrained multilingual embeddings;
- add systematic error analysis by document length and inferred source;
- expose the selected Text bundle through a small local API or command-line batch predictor.

## Reproducibility statement

Random state 42 is used wherever supported. All Text feature fitting and hyperparameter selection occur without access to the held-out test partition. Clustering receives no language labels during representation learning, parameter selection, or fitting. Source data remains read-only and all generated artifacts are stored separately.
