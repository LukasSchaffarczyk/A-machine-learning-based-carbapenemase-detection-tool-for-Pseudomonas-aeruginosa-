from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

DATA_FILE = "data.xlsx"
OUTPUT_DIR = "model_output"
TARGET_COLUMN = "Carbapenemase"
NEGATIVE_LABEL = "negative"
TARGET_SENSITIVITY = 0.95
SPLIT_FRACTION = 0.15
RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 1000
CI_LEVEL = 0.95
SKIPROWS = 1

complete_panel = ["PIP", "TZP", "TAZ", "CZA", "CET", "FEP", "CID", "FPE", "ATZ", "AZA", "IMI", "IMR", "MEM", "MEV", "CIP", "AK"]
standard_panel = ["PIP", "TZP", "TAZ", "FEP", "ATZ", "IMI", "MEM", "CIP", "AK"]
novel_betalactam_panel = ["CET", "CZA", "CID", "AZA", "IMR", "MEV", "FPE"]
minimal_panel = ["FEP", "ATZ", "IMI", "MEM", "IMR", "CET"]

FEATURE_SETS = {
    "complete_panel": standard_panel + novel_betalactam_panel,
    "standard_panel": standard_panel,
    "novel_betalactam_panel": novel_betalactam_panel,
    "minimal_panel": minimal_panel,
}

CATBOOST_PARAMS = {
    "task_type": "CPU",
    "depth": 7,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "l2_leaf_reg": 1e-5,
    "random_strength": 0.8,
    "one_hot_max_size": 15,
    "boosting_type": "Plain",
    "bootstrap_type": "Bernoulli",
    "leaf_estimation_iterations": 1,
    "n_estimators": 1000,
    "max_bin": 254,
    "od_type": "Iter",
    "od_wait": 300,
}


def clean_numeric(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"n.d.", "nd", "nan"}:
            return np.nan
        if value.endswith("d"):
            value = value[:-1]
        if value.startswith(("≥", "≤")):
            value = value[1:]
        elif value.startswith((">=", "<=")):
            value = value[2:]
        value = value.replace(",", ".")
    return float(value)


def prepare_dataset(df):
    df = df.copy()
    df["target"] = df[TARGET_COLUMN].apply(
        lambda x: 0 if str(x).strip().lower() == NEGATIVE_LABEL else 1
    )
    for column in complete_panel:
        df[column] = df[column].apply(clean_numeric)
    return df


def load_datasets():
    internal = prepare_dataset(pd.read_excel(DATA_FILE, sheet_name="Internal Dataset", skiprows=SKIPROWS))
    external_2 = prepare_dataset(pd.read_excel(DATA_FILE, sheet_name="External_Centre2", skiprows=SKIPROWS))
    external_3 = prepare_dataset(pd.read_excel(DATA_FILE, sheet_name="External_Centre3", skiprows=SKIPROWS))
    return internal, external_2, external_3


def sensitivity(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred != 1))
    return tp / (tp + fn)


def specificity(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred != 0))
    return tn / (tn + fp)


def bootstrap_ci(y, probabilities, metric, threshold=None):
    y = np.asarray(y)
    probabilities = np.asarray(probabilities)
    rng = np.random.default_rng(RANDOM_SEED)
    values = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, len(y), len(y))
        y_boot = y[indices]
        p_boot = probabilities[indices]
        if metric == "auroc":
            if len(np.unique(y_boot)) == 2:
                values.append(roc_auc_score(y_boot, p_boot))
        elif metric == "sensitivity":
            if np.any(y_boot == 1):
                values.append(sensitivity(y_boot, p_boot >= threshold))
        elif metric == "specificity":
            if np.any(y_boot == 0):
                values.append(specificity(y_boot, p_boot >= threshold))

    if metric == "auroc":
        point = roc_auc_score(y, probabilities)
    elif metric == "sensitivity":
        point = sensitivity(y, probabilities >= threshold)
    else:
        point = specificity(y, probabilities >= threshold)

    alpha = (1 - CI_LEVEL) / 2
    lower, upper = np.quantile(values, [alpha, 1 - alpha])
    return float(point), float(lower), float(upper)


def threshold_for_target_sensitivity(y, probabilities):
    fpr, tpr, thresholds = roc_curve(y, probabilities)
    index = np.where(tpr >= TARGET_SENSITIVITY)[0][0]
    return float(thresholds[index]), float(tpr[index]), float(1 - fpr[index])


def evaluate(name, data, model, features, threshold):
    y = data["target"].to_numpy()
    probabilities = model.predict_proba(Pool(data[features], cat_features=[]))[:, 1]
    predictions = probabilities >= threshold
    tp = int(np.sum((y == 1) & (predictions == 1)))
    tn = int(np.sum((y == 0) & (predictions == 0)))
    fp = int(np.sum((y == 0) & (predictions == 1)))
    fn = int(np.sum((y == 1) & (predictions == 0)))
    auroc, auroc_low, auroc_high = bootstrap_ci(y, probabilities, "auroc")
    sens, sens_low, sens_high = bootstrap_ci(y, probabilities, "sensitivity", threshold)
    spec, spec_low, spec_high = bootstrap_ci(y, probabilities, "specificity", threshold)

    print(f"{name}: n={len(data)}, positives={int(y.sum())}, negatives={int((y == 0).sum())}")
    print(f"  AUROC {auroc:.3f} (95% CI {auroc_low:.3f}-{auroc_high:.3f})")
    print(f"  Fixed threshold {threshold:.6f}: sensitivity {sens:.3f} ({sens_low:.3f}-{sens_high:.3f}), specificity {spec:.3f} ({spec_low:.3f}-{spec_high:.3f})")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    return {
        "dataset": name,
        "n": len(data),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "threshold": threshold,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "auroc": auroc,
        "auroc_ci_low": auroc_low,
        "auroc_ci_high": auroc_high,
        "sensitivity": sens,
        "sensitivity_ci_low": sens_low,
        "sensitivity_ci_high": sens_high,
        "specificity": spec,
        "specificity_ci_low": spec_low,
        "specificity_ci_high": spec_high,
    }


def train_model(feature_set, features, internal, external_2, external_3):
    development, test, y_development, y_test = train_test_split(
        internal,
        internal["target"],
        test_size=SPLIT_FRACTION,
        random_state=RANDOM_SEED,
        stratify=internal["target"],
    )

    train, validation, y_train, y_validation = train_test_split(
        development,
        y_development,
        test_size=SPLIT_FRACTION / (1 - SPLIT_FRACTION),
        random_state=RANDOM_SEED,
        stratify=y_development,
    )

    print(f"\n=== {feature_set} ===")
    print(f"Input features: {features}")
    print(f"Training set: {len(train)} | Validation set: {len(validation)} | Test set: {len(test)}")

    model = CatBoostClassifier(
        **CATBOOST_PARAMS,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        verbose=100,
        devices="0",
    )

    model.fit(
        Pool(train[features], label=y_train, cat_features=[]),
        eval_set=Pool(validation[features], label=y_validation, cat_features=[]),
    )

    test_probabilities = model.predict_proba(Pool(test[features], cat_features=[]))[:, 1]
    threshold, observed_sensitivity, observed_specificity = threshold_for_target_sensitivity(
        y_test.to_numpy(), test_probabilities
    )

    print(
        f"Threshold calibrated on Internal Test for target sensitivity {TARGET_SENSITIVITY:.2f}: "
        f"{threshold:.6f} (observed sensitivity {observed_sensitivity:.3f}, specificity {observed_specificity:.3f})"
    )

    results = [
        evaluate("Internal_Test", test, model, features, threshold),
        evaluate("External_Centre2", external_2, model, features, threshold),
        evaluate("External_Centre3", external_3, model, features, threshold),
    ]

    for result in results:
        result["feature_set"] = feature_set

    return model, threshold, results


def main():
    internal, external_2, external_3 = load_datasets()

    print(f"Internal Dataset: n={len(internal)}, positive={int(internal.target.sum())}, negative={int((internal.target == 0).sum())}")
    print(f"External_Centre2: n={len(external_2)}, positive={int(external_2.target.sum())}, negative={int((external_2.target == 0).sum())}")
    print(f"External_Centre3: n={len(external_3)}, positive={int(external_3.target.sum())}, negative={int((external_3.target == 0).sum())}")

    models = {}
    thresholds = {}
    results = []

    for feature_set, features in FEATURE_SETS.items():
        model, threshold, feature_results = train_model(
            feature_set, features, internal, external_2, external_3
        )
        models[feature_set] = model
        thresholds[feature_set] = threshold
        results.extend(feature_results)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature_set, model in models.items():
        model.save_model(output_dir / f"model_{feature_set}.cbm")

    with open(output_dir / "thresholds.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                feature_set: {
                    "threshold": thresholds[feature_set],
                    "target_sensitivity": TARGET_SENSITIVITY,
                    "calibration_dataset": "Internal Test",
                }
                for feature_set in FEATURE_SETS
            },
            file,
            indent=2,
        )

    pd.DataFrame(results).to_csv(output_dir / "results.csv", index=False)
    print(f"\nCompleted. Models, thresholds and results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
