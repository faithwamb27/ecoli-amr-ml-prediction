# Machine Learning Prediction of Antibiotic Resistance in *Escherichia coli*

## Project Overview

This project uses public *Escherichia coli* genomic and antibiotic susceptibility data to predict antimicrobial resistance from accessory gene presence/absence patterns. The main objective is to evaluate whether machine learning can distinguish resistant and susceptible bacterial isolates and whether predictive performance differs across antibiotics with different biological resistance mechanisms.

The analysis focuses on a supervised binary classification task:

**Can an *E. coli* isolate be predicted as resistant or susceptible to an antibiotic using gene presence/absence data?**

This project combines bioinformatics, microbial genetics, and machine learning to study antimicrobial resistance prediction using publicly available data.

## Biological Motivation

Antimicrobial resistance is a major public health challenge. In *E. coli*, resistance can arise through several mechanisms, including acquisition of resistance genes, beta-lactamase activity, mobile genetic elements, efflux pumps, and chromosomal mutations.

Gene presence/absence data are especially useful for studying resistance mechanisms driven by acquired genes, such as beta-lactamase-mediated resistance. However, this feature type may be less informative for resistance mechanisms driven primarily by point mutations, such as some fluoroquinolone resistance phenotypes.

This project evaluates that biological distinction by comparing model performance across multiple antibiotics.

## Dataset

This starter project uses two public CSV files:

1. `Metadata.csv`
   - Isolate ID
   - MLST sequence type
   - Year of isolation
   - Antibiotic susceptibility labels for 12 drugs

2. `AccessoryGene.csv`
   - Gene presence/absence matrix
   - 0 = gene absent
   - 1 = gene present

The code downloads both files directly from the public GitHub dataset.

## Antibiotics available

| Code | Antibiotic |
|---|---|
| CTZ | Ceftazidime |
| CTX | Cefotaxime |
| CXM | Cefuroxime |
| CET | Cephalothin |
| AMP | Ampicillin |
| AMX | Amoxicillin |
| AMC | Amoxicillin + clavulanate |
| TZP | Piperacillin/tazobactam |
| GEN | Gentamicin |
| TBM | Tobramycin |
| TMP | Trimethoprim |
| CIP | Ciprofloxacin |

## How to run

Install requirements:

```bash
pip install -r requirements.txt
```

Run the model for a specific antibiotic (default is ciprofloxacin):

```bash
python amr_ecoli_model.py --drug CIP
```

Or specify a different antibiotic:

```bash
python amr_ecoli_model.py --drug AMP
python amr_ecoli_model.py --drug CTX
```

Customize the output directory:

```bash
python amr_ecoli_model.py --drug CIP --outdir my_results
```

## Output files

For each run, the script generates:

- `{drug}_confusion_matrix.png` - Visualization of model performance
- `{drug}_test_predictions.csv` - Individual predictions on test set
- `{drug}_metrics.csv` - ROC-AUC, accuracy, and sample counts
- `{drug}_feature_importance_top_30.csv` - Top 30 most important genes
- `{drug}_feature_importance_all.csv` - All feature importances
- `{drug}_top_feature_importance.png` - Bar plot of top features
- `{drug}_random_forest_model.joblib` - Trained model for deployment
- `ecoli_amr_merged_metadata_gene_presence.csv` - Full merged dataset

## Model details

- **Algorithm**: Random Forest Classifier (400 trees)
- **Features**: Gene presence/absence patterns
- **Train/Test split**: 80/20 with MLST-aware stratification
- **Class weighting**: Balanced (handles imbalanced datasets)
- **Feature selection**: Variance threshold to remove zero-variance features

## Citation

Dataset source: [Lucy-Moctezuma/ML-Tutorial-for-Antibiotic-Resistance-Predictions-for-E.-Coli](https://github.com/Lucy-Moctezuma/ML-Tutorial-for-Antibiotic-Resistance-Predictions-for-E.-Coli)

## License

This repository is released under the MIT License. The license applies to the code and analysis in this repository. The original dataset remains subject to the terms of its source repository.

## Author

Faith Wanjiku Wambui


