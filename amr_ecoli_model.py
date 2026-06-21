"""
AMR E. coli Machine Learning Starter Project
Predict antibiotic resistance using E. coli gene presence/absence data.

Dataset source:
- Metadata.csv
- AccessoryGene.csv
from Lucy-Moctezuma/ML-Tutorial-for-Antibiotic-Resistance-Predictions-for-E.-Coli

Run:
    python amr_ecoli_model.py --drug CIP

Common drug options:
    CTZ = ceftazidime
    CTX = cefotaxime
    CXM = cefuroxime
    CET = cephalothin
    AMP = ampicillin
    AMX = amoxicillin
    AMC = amoxicillin + clavulanate
    TZP = piperacillin/tazobactam
    GEN = gentamicin
    TBM = tobramycin
    TMP = trimethoprim
    CIP = ciprofloxacin
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline


METADATA_URL = (
    "https://raw.githubusercontent.com/Lucy-Moctezuma/"
    "ML-Tutorial-for-Antibiotic-Resistance-Predictions-for-E.-Coli/"
    "main/Datasets/Metadata.csv"
)

GENE_PRESENCE_URL = (
    "https://raw.githubusercontent.com/Lucy-Moctezuma/"
    "ML-Tutorial-for-Antibiotic-Resistance-Predictions-for-E.-Coli/"
    "main/Datasets/AccessoryGene.csv"
)

ANTIBIOTIC_COLUMNS = [
    "CTZ", "CTX", "CXM", "CET", "AMP", "AMX",
    "AMC", "TZP", "GEN", "TBM", "TMP", "CIP"
]

DRUG_NAMES = {
    "CTZ": "ceftazidime",
    "CTX": "cefotaxime",
    "CXM": "cefuroxime",
    "CET": "cephalothin",
    "AMP": "ampicillin",
    "AMX": "amoxicillin",
    "AMC": "amoxicillin + clavulanate",
    "TZP": "piperacillin/tazobactam",
    "GEN": "gentamicin",
    "TBM": "tobramycin",
    "TMP": "trimethoprim",
    "CIP": "ciprofloxacin",
}


def load_data() -> pd.DataFrame:
    """Load metadata and gene presence/absence tables, then merge by isolate."""
    print("Loading metadata...")
    metadata = pd.read_csv(METADATA_URL)

    print("Loading gene presence/absence matrix...")
    gene_presence = pd.read_csv(GENE_PRESENCE_URL)

    # The first column in the gene file is the isolate identifier.
    first_col = gene_presence.columns[0]
    gene_presence = gene_presence.rename(columns={first_col: "Isolate"})

    merged = pd.merge(metadata, gene_presence, on="Isolate", how="inner")

    if "Lane.accession" in merged.columns:
        merged = merged.drop(columns=["Lane.accession"])

    print(f"Merged dataset shape: {merged.shape}")
    return merged


def prepare_binary_dataset(df: pd.DataFrame, drug: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Prepare X, y, and MLST groups for one antibiotic."""
    drug = drug.upper()
    if drug not in ANTIBIOTIC_COLUMNS:
        raise ValueError(f"{drug} is not recognized. Use one of: {ANTIBIOTIC_COLUMNS}")

    # Keep rows with phenotype labels for this drug.
    work = df.dropna(subset=[drug]).copy()

    # Treat Intermediate as resistant, following the tutorial's approach.
    label_map = {
        "S": 0,
        "Susceptible": 0,
        "susceptible": 0,
        "R": 1,
        "Resistant": 1,
        "resistant": 1,
        "I": 1,
        "Intermediate": 1,
        "intermediate": 1,
    }
    work["target"] = work[drug].map(label_map)
    work = work.dropna(subset=["target"]).copy()
    work["target"] = work["target"].astype(int)

    # Exclude identifiers, labels, and MLST. Use year + gene presence/absence features.
    exclude = set(ANTIBIOTIC_COLUMNS + ["target", "Isolate", "MLST"])
    feature_cols = [c for c in work.columns if c not in exclude]

    X = work[feature_cols].copy()

    # Make sure all features are numeric.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Fill missing numeric values with the median, then with 0 if median is missing.
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    y = work["target"].copy()

    if "MLST" in work.columns:
        groups = work["MLST"].fillna("missing_MLST").astype(str)
    else:
        groups = pd.Series(np.arange(len(work)), index=work.index)

    print(f"\nDrug: {drug} ({DRUG_NAMES.get(drug, drug)})")
    print(f"Samples with usable phenotype labels: {len(work)}")
    print("Class balance:")
    print(y.value_counts().rename(index={0: "Susceptible", 1: "Resistant"}))

    return X, y, groups, work


def split_data(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    """
    Split the data.

    Preferred: group split by MLST so related isolates are not split between
    training and testing. Fallback: regular stratified split.
    """
    try:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        print("Used GroupShuffleSplit by MLST.")
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    except Exception as exc:
        warnings.warn(f"Group split failed, using stratified random split. Reason: {exc}")
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a random forest model."""
    model = Pipeline(
        steps=[
            ("variance", VarianceThreshold(threshold=0.0)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                    max_features="sqrt",
                ),
            ),
        ]
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, outdir: Path, drug: str) -> pd.DataFrame:
    """Evaluate model and save metrics/figures."""
    outdir.mkdir(parents=True, exist_ok=True)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    report = classification_report(
        y_test,
        pred,
        target_names=["Susceptible", "Resistant"],
        zero_division=0,
    )
    print(report)

    try:
        auc = roc_auc_score(y_test, proba)
        print(f"ROC-AUC: {auc:.3f}")
    except Exception:
        auc = np.nan
        print("ROC-AUC could not be calculated.")

    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Susceptible", "Resistant"],
    )
    disp.plot()
    plt.title(f"Confusion Matrix: {drug}")
    plt.tight_layout()
    plt.savefig(outdir / f"{drug}_confusion_matrix.png", dpi=300)
    plt.close()

    predictions = pd.DataFrame(
        {
            "true_label": y_test.values,
            "predicted_label": pred,
            "predicted_resistance_probability": proba,
        },
        index=y_test.index,
    )
    predictions.to_csv(outdir / f"{drug}_test_predictions.csv", index=True)

    metrics = pd.DataFrame(
        {
            "drug": [drug],
            "drug_name": [DRUG_NAMES.get(drug, drug)],
            "n_test": [len(y_test)],
            "roc_auc": [auc],
            "accuracy": [(pred == y_test).mean()],
        }
    )
    metrics.to_csv(outdir / f"{drug}_metrics.csv", index=False)

    return predictions


def save_feature_importance(model: Pipeline, X: pd.DataFrame, outdir: Path, drug: str, top_n: int = 30) -> pd.DataFrame:
    """Save top model features."""
    variance = model.named_steps["variance"]
    rf = model.named_steps["model"]

    kept_features = np.array(X.columns)[variance.get_support()]
    importances = rf.feature_importances_

    importance_df = (
        pd.DataFrame({"feature": kept_features, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    importance_df.to_csv(outdir / f"{drug}_feature_importance_all.csv", index=False)
    top = importance_df.head(top_n)
    top.to_csv(outdir / f"{drug}_feature_importance_top_{top_n}.csv", index=False)

    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.xlabel("Random forest feature importance")
    plt.title(f"Top {top_n} predictors for {drug} resistance")
    plt.tight_layout()
    plt.savefig(outdir / f"{drug}_top_feature_importance.png", dpi=300)
    plt.close()

    print(f"\nTop {top_n} features:")
    print(top)

    return importance_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", default="CIP", help="Antibiotic abbreviation, e.g., CIP, AMP, CTX")
    parser.add_argument("--outdir", default="outputs", help="Directory to save outputs")
    args = parser.parse_args()

    drug = args.drug.upper()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    merged = load_data()
    merged.to_csv(outdir / "ecoli_amr_merged_metadata_gene_presence.csv", index=False)

    X, y, groups, work = prepare_binary_dataset(merged, drug=drug)
    X_train, X_test, y_train, y_test = split_data(X, y, groups)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test, outdir, drug)
    save_feature_importance(model, X, outdir, drug)

    joblib.dump(model, outdir / f"{drug}_random_forest_model.joblib")
    print(f"\nDone. Results saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
