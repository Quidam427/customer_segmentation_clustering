# Customer Segmentation for an Online Retailer

AIM Capstone Project (Clustering) by Frederick Halog.

Two years of sales from a UK online gift shop are used to sort its 5,852 customers into four groups based on how they shop, so that each group can be marketed to differently.

## Results at a glance

| Segment | Customers | Share of sales | Typical customer |
|---|---|---|---|
| Loyal Regulars | 2,137 (37%) | 83% | 9 orders, £2,976 spent, last order a month ago |
| New Customers | 1,689 (29%) | 7% | 2 orders, £542 spent, first bought about six months ago |
| Lapsed Customers | 1,825 (31%) | 6% | 1 order, £358 spent, last seen over a year ago |
| Heavy Returners | 201 (3%) | 4% | Send back 37% of what they buy |

- **Chosen model:** K-Means with 4 clusters, Silhouette Score 0.265. It was compared with Hierarchical clustering (Ward, 0.215) and DBSCAN, which could not separate the customers into usable groups.
- **Stability:** the same groups come back across 10 random starts (Adjusted Rand Index 0.99) and on a hidden 30% of customers.
- **Fairness:** UK and non-UK customers reach the Loyal Regulars segment at similar rates (disparate impact ratio 0.94). At the same level of spending, however, non-UK customers are placed there less often, because they place fewer, larger orders. A spend-based rule that narrows this gap is tested in the notebook.

The Silhouette Score is modest. The customers form a continuous range, not separate islands, so the segments are best read as zones along a spectrum.

## Repository layout

```
├── notebooks/        Main notebook: the full analysis, Steps 1 to 5
├── src/              The same data preparation and segment assignment as plain Python scripts
├── data/
│   ├── raw/          Original Excel file from UCI
│   └── processed/    Customer features and segment assignments produced by the notebook
├── models/           Trained models, settings, score tables and the fairness audit
├── presentations/    Technical deck (Jupyter slides) and business deck (PowerPoint)
├── requirements.txt  Python packages and versions
└── LICENSE           MIT license for the code
```

## What is in the notebook

[notebooks/Frederick_Halog_AIM Capstone_Clustering.ipynb](notebooks/Frederick_Halog_AIM%20Capstone_Clustering.ipynb) follows the capstone steps and is written for readers without a data science background.

| Step | Content |
|---|---|
| 1. Problem framing | Business problem, task type, success measures |
| 2. Data understanding | Data quality checks and a data dictionary |
| 3. Preprocessing, EDA and feature engineering | Cleaning, RFM and other customer features, outlier treatment, feature selection, PCA, t-SNE and UMAP |
| 4. Modelling | K-Means, Hierarchical and DBSCAN, compared on Silhouette, Davies-Bouldin and Calinski-Harabasz |
| 5. Ethics and bias audit | SHAP, PDP and ICE explanations, limitations, fairness audit of UK against non-UK customers |

## How to reproduce

Requires Python 3.14 (the version this was built and tested on).

```bash
git clone https://github.com/Quidam427/customer_segmentation_clustering.git
cd customer_segmentation_clustering
python -m venv .venv
.venv\Scripts\activate            # on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook in `notebooks/` and choose **Run All**. The first run reads the Excel file, which takes a few minutes, and saves a fast-loading copy for later runs. A full run takes roughly ten minutes.

The random seed is fixed at 42, so the results are the same on every run.

### Running the pipeline as a script

Once the notebook has been run (or using the models already saved in `models/`), the data preparation and segment assignment can be repeated without the notebook:

```bash
python -m src.run_pipeline
```

This rebuilds the customer features from the raw data, places every customer in a segment using the saved model, and reports how closely the result matches the notebook (it should be 100%).

| File | Purpose |
|---|---|
| `src/config.py` | File locations and the random seed |
| `src/data_prep.py` | Load and clean the data, build customer features, prepare them for the model |
| `src/segment.py` | Load the saved model and assign segments |
| `src/run_pipeline.py` | Runs everything end to end |

## Presentations

| Audience | File |
|---|---|
| Technical (peers) | [presentations/Technical_Presentation.slides.html](presentations/Technical_Presentation.slides.html). Download it and open it in a browser. Built from `Technical_Presentation.ipynb`. |
| Business (executives) | [presentations/Business_Presentation.pptx](presentations/Business_Presentation.pptx) |

To rebuild the technical slides after editing their notebook:

```bash
cd presentations
jupyter nbconvert --to slides --no-input --SlidesExporter.reveal_scroll=True Technical_Presentation.ipynb
```

## Data source and citation

The data is the **Online Retail II** dataset from the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/502/online+retail+ii

It is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0), and a copy is included in `data/raw/`.

> Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

> Chen, D., Sain, S. L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management*, 19(3), 197–208.

## license

The code in this repository is released under the MIT license (see `LICENSE`). The dataset keeps its own CC BY 4.0 license.

## Limitations

- Segments overlap at the edges, so a segment label is a guide, not a fixed fact about a customer.
- About 15% of sales value has no customer number (mostly UK guest checkouts) and is not covered.
- The segments are a snapshot as of 10 December 2011 and would need refreshing regularly.
- Fairness could only be checked by country. The data contains no age, gender or ethnicity information.
