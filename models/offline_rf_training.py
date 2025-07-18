import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

# ==================================================
# Global Options for Final Offline RF Training & Holdout Evaluation

# Default dataset path from your previous offline experiments
default_dataset_path = "/Users/omar/Desktop/application/First year/Scripts/BCCC-CIC-Bell-DNS-2024/Test/oracle_psuedo_label_dataset.csv"

# Choose target method:
#   "aggregated"  -> Aggregate multiple pseudo-label columns using per-datapoint agreement.
#   "weighted"    -> Compute a weighted combination of anomaly scores.
target_method = "aggregated"  # Options: "aggregated" or "weighted"

# For aggregated target, list all pseudo-label columns from different methods:
pseudo_label_columns = [
    "hybrid_anomaly_label", "oracle_psuedo_label",
    "lof_anomaly_label_withKM",
    "ocsvm_anomaly_label_withKM", "ocsvm_anomaly_label_noKM"
]
# Name of the aggregated pseudo-label column:
aggregated_label_column = "aggregated_label"

# For aggregated target, set the list of minimum agreement thresholds to try (e.g., [2, 3, 4])
min_agreement_options = [2, 3]

# For weighted target, define weights and threshold:
weighted_hybrid = 0.5
weighted_ocsvm_noKM = 0.3
weighted_ocsvm_withKM = 0.2
weighted_threshold = 0.5  # Threshold for binarizing weighted score

# Excluded columns for RF training – these columns are omitted from features.
excluded_columns = [
    "label", "final_label", "hybrid_score", "cluster_label", "oracle_psuedo_label",
    "kmeans_score", "kmeans_anomaly_label", "if_score", "hybrid_anomaly_label",
    "lof_score_withKM", "lof_anomaly_label_withKM", "lof_score_noKM", "lof_anomaly_label_noKM",
    "ocsvm_score_withKM", "ocsvm_anomaly_label_withKM", "ocsvm_score_noKM", "ocsvm_anomaly_label_noKM",
    "aggregated_label", "weighted_target"
]

# Cross-validation fold settings – try 3-, 5-, 7-, and 9-fold CV by default.
cv_folds_options = [3, 5]

# Reserve 20% of the data as a holdout set.
holdout_fraction = 0.2

# ==================================================
# RF hyperparameters for final training and cross validation:
rf_params = {
    'n_estimators': 400,
    'max_depth': 30,
    'max_features': 0.5,
    'min_samples_leaf': 1,
    'criterion': 'entropy',
    'random_state': 42
}

# ==================================================
def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} does not exist.")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise IOError(f"Error reading file {file_path}: {e}")
    return df

def validate_dataset(df, required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

def aggregate_pseudo_labels(df, pseudo_cols, min_agreement):
    """Aggregate multiple pseudo-label columns using majority vote.
       For each row, if one label (0 or 1) appears at least min_agreement times, assign that label;
       otherwise, assign -1 (uncertain)."""
    aggregated = []
    for _, row in df.iterrows():
        votes = []
        for col in pseudo_cols:
            try:
                vote = int(row[col])
                if vote in [0, 1]:
                    votes.append(vote)
            except:
                continue
        if len(votes) == 0:
            aggregated.append(-1)
        else:
            count0 = votes.count(0)
            count1 = votes.count(1)
            if count0 >= min_agreement and count0 > count1:
                aggregated.append(0)
            elif count1 >= min_agreement and count1 > count0:
                aggregated.append(1)
            else:
                aggregated.append(-1)
    df[aggregated_label_column] = aggregated
    return df

def compute_weighted_target(df, weighted_hybrid, weighted_ocsvm_noKM, weighted_ocsvm_withKM, threshold):
    """Compute a weighted target from anomaly scores and binarize using the threshold."""
    for col in ["hybrid_score", "ocsvm_score_noKM", "ocsvm_score_withKM"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' required for weighted target not found.")
    weighted_score = (weighted_hybrid * df["hybrid_score"] +
                      weighted_ocsvm_noKM * df["ocsvm_score_noKM"] +
                      weighted_ocsvm_withKM * df["ocsvm_score_withKM"])
    weighted_target = (weighted_score >= threshold).astype(int)
    df["weighted_target"] = weighted_target
    return df

def build_feature_matrix(df, excluded_cols):
    features = [col for col in df.columns if col not in excluded_cols]
    if not features:
        raise ValueError("No features available after applying the excluded columns list.")
    X = df[features].copy()
    return X, features

def build_target_vector(df, target_method, aggregated_col="aggregated_label", weighted_col="weighted_target"):
    if target_method == "aggregated":
        if aggregated_col not in df.columns:
            raise ValueError(f"Aggregated target column '{aggregated_col}' not found.")
        Y = df[aggregated_col].copy()
        target_name = aggregated_col
    elif target_method == "weighted":
        if weighted_col not in df.columns:
            raise ValueError(f"Weighted target column '{weighted_col}' not found.")
        Y = df[weighted_col].copy()
        target_name = weighted_col
    else:
        raise ValueError("target_method must be either 'aggregated' or 'weighted'.")
    return Y, target_name

def compute_metrics(gt, pred):
    mask = (pred != -1)
    if mask.sum() == 0:
        return {"Accuracy": None, "Precision": None, "Recall": None, "F1": None, 
                "Accuracy(0)": None, "Accuracy(1)": None, "ARI": None}
    gt_valid = gt[mask]
    pred_valid = pred[mask]
    metrics = {
        "Accuracy": accuracy_score(gt_valid, pred_valid),
        "Precision": precision_score(gt_valid, pred_valid, zero_division=0),
        "Recall": recall_score(gt_valid, pred_valid, zero_division=0),
        "F1": f1_score(gt_valid, pred_valid, zero_division=0),
        "ARI": adjusted_rand_score(gt_valid, pred_valid)
    }
    mask0 = (gt_valid == 0)
    mask1 = (gt_valid == 1)
    metrics["Accuracy(0)"] = accuracy_score(gt_valid[mask0], pred_valid[mask0]) if mask0.sum() > 0 else None
    metrics["Accuracy(1)"] = accuracy_score(gt_valid[mask1], pred_valid[mask1]) if mask1.sum() > 0 else None
    return metrics

def cross_validate_rf(X, Y, cv_folds, rf_params):
    results = []
    for folds in cv_folds:
        print(f"\nPerforming {folds}-fold cross validation...")
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        fold_metrics = []
        fold_idx = 1
        for train_idx, test_idx in skf.split(X, Y):
            X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
            Y_train_cv, Y_test_cv = Y.iloc[train_idx], Y.iloc[test_idx]
            model = RandomForestClassifier(**rf_params)
            model.fit(X_train_cv, Y_train_cv)
            Y_pred_cv = model.predict(X_test_cv)
            metrics = compute_metrics(Y_test_cv.values, Y_pred_cv)
            print(f" Fold {fold_idx}:")
            for key, val in metrics.items():
                print(f"   {key}: {val}")
            fold_metrics.append(metrics)
            fold_idx += 1
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            valid_vals = [m[key] for m in fold_metrics if m[key] is not None]
            avg_metrics[key] = np.mean(valid_vals) if valid_vals else None
        results.append({
            "cv_folds": folds,
            **avg_metrics
        })
    return results

def print_dict_metrics(metrics, title="Metrics"):
    print(f"\n{title}:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")

# -----------------------------------------------------------
# Run pipeline for a given min_agreement (if aggregated) and CV fold configuration,
# including splitting off a holdout set (20%) for final evaluation.
def run_pipeline(min_agreement=None, cv_folds_val=None):
    # Load and validate dataset
    df = load_dataset(default_dataset_path)
    required_cols = ["label", "hybrid_score", "lof_score_withKM", "lof_score_noKM",
                     "ocsvm_score_withKM", "ocsvm_score_noKM"]
    if target_method == "aggregated":
        required_cols.extend(pseudo_label_columns)
    validate_dataset(df, required_cols)
    
    # Generate target based on chosen method.
    if target_method == "aggregated":
        print(f"\nAggregating pseudo-labels with minimum agreement >= {min_agreement} ...")
        df = aggregate_pseudo_labels(df, pseudo_label_columns, min_agreement)
        # Print aggregated pseudo label distribution for insight:
        print("\nAggregated pseudo label distribution:")
        print(df[aggregated_label_column].value_counts())
        target_used = aggregated_label_column
    elif target_method == "weighted":
        print("\nComputing weighted target from anomaly scores ...")
        df = compute_weighted_target(df, weighted_hybrid, weighted_ocsvm_noKM, weighted_ocsvm_withKM, weighted_threshold)
        target_used = "weighted_target"
    else:
        raise ValueError("Invalid target_method specified.")
    
    # Compute training coverage: percentage of data points with confident pseudo labels (0 or 1)
    training_coverage = (df[df[target_used].isin([0, 1])].shape[0] / df.shape[0]) * 100
    print(f"\nTraining coverage: {training_coverage:.2f}% of total data points have high-confidence pseudo labels.")
    
    # ... (the rest of your pipeline remains unchanged)

    # Use only rows with target in {0,1} (clean training set)
    df_clean = df[df[target_used].isin([0, 1])].copy()
    if df_clean.empty:
        raise ValueError("No samples with target in {0,1} found. Cannot train RF model.")
    print(f"\nNumber of samples used for training ({target_used} in {{0,1}}): {len(df_clean)}")
    
    # Split df_clean into training (80%) and holdout (20%) sets (stratified by target)
    train_df, holdout_df = train_test_split(df_clean, test_size=holdout_fraction, random_state=42, stratify=df_clean[target_used])
    print(f"Training set: {len(train_df)} samples; Holdout set: {len(holdout_df)} samples")
    
    # Build feature matrix for training set and print features used
    X_train, features_used = build_feature_matrix(train_df, excluded_columns)
    Y_train, _ = build_target_vector(train_df, target_method, aggregated_label_column, "weighted_target")
    
    print("\nFeatures used for training:")
    print(features_used)
    print("Using target:", target_used)
    
    # Run cross-validation on training set using specified cv_folds_val
    cv_results = cross_validate_rf(X_train, Y_train, [cv_folds_val], rf_params)
    
    # Train final RF model on entire training set
    final_rf = RandomForestClassifier(**rf_params)
    final_rf.fit(X_train, Y_train)
    
    # Build feature matrix for holdout set
    X_holdout, _ = build_feature_matrix(holdout_df, excluded_columns)
    Y_holdout, _ = build_target_vector(holdout_df, target_method, aggregated_label_column, "weighted_target")
    
    # Evaluate final model on holdout set against ground truth labels (from 'label' column)
    holdout_pred = final_rf.predict(X_holdout)
    holdout_metrics = compute_metrics(Y_holdout.values, holdout_pred)
    
    # Evaluate training target (aggregated_label) against ground truth labels in training set
    GT_train = train_df["label"]
    train_target_metrics = compute_metrics(GT_train.values, Y_train.values)
    
    result = {
        "target_method": target_method,
        "min_agreement": min_agreement if target_method == "aggregated" else None,
        "cv_folds": cv_folds_val,
        "training_coverage": training_coverage,
        "cv_results": cv_results[0],
        "holdout_metrics": holdout_metrics,
        "train_target_vs_GT": train_target_metrics,
        "final_rf_model": final_rf  # Return the final RF model so it can be saved
    }
    return result

def main():
    model_path = "best_offline_rf_model.joblib"
    # If the best model file already exists, do not run the training pipeline.
    if os.path.exists(model_path):
        print(f"File '{model_path}' already exists. Skipping training to avoid redundancy.")
        return

    results = []
    # Loop over the combinations of min_agreement and CV folds.
    if target_method == "aggregated":
        for min_agree in min_agreement_options:
            for folds in cv_folds_options:
                print("\n" + "="*50)
                print(f"Running pipeline with min_agreement = {min_agree} and {folds}-fold CV")
                try:
                    res = run_pipeline(min_agree, folds)
                    results.append(res)
                except Exception as e:
                    print(f"Error with min_agreement={min_agree}, cv_folds={folds}: {e}")
    elif target_method == "weighted":
        for folds in cv_folds_options:
            print("\n" + "="*50)
            print(f"Running pipeline with weighted target and {folds}-fold CV")
            try:
                res = run_pipeline(None, folds)
                results.append(res)
            except Exception as e:
                print(f"Error with weighted target, cv_folds={folds}: {e}")
    else:
        print("Invalid target_method specified.")
        return

    # Print summary of results:
    for res in results:
        print("\n" + "-"*40)
        print(f"Target Method: {res['target_method']}")
        if res['target_method'] == "aggregated":
            print(f"Min Agreement: {res['min_agreement']}")
        print(f"CV Folds: {res['cv_folds']}")
        print(f"Training Coverage: {res['training_coverage']:.2f}%")
        print("CV Results:")
        for k, v in res['cv_results'].items():
            print(f"  {k}: {v}")
        print("Holdout Metrics:")
        for k, v in res['holdout_metrics'].items():
            print(f"  {k}: {v}")
        print("Training Target vs. GT Metrics:")
        for k, v in res['train_target_vs_GT'].items():
            print(f"  {k}: {v}")

    # Now select the best model based on both holdout accuracy and training coverage.
    # The selection is done by first finding the maximum training coverage; then, among those experiments,
    # the one with the highest holdout accuracy is chosen.
    best_coverage = max(res["training_coverage"] for res in results)
    candidates = [res for res in results if res["training_coverage"] == best_coverage]
    
    best_model = None
    best_acc = -1
    for res in candidates:
        holdout_acc = res["holdout_metrics"].get("Accuracy", 0)
        if holdout_acc is not None and holdout_acc > best_acc:
            best_acc = holdout_acc
            best_model = res["final_rf_model"]
    
    if best_model is not None:
        joblib.dump(best_model, model_path)
        print(f"\nBest offline RF model saved to '{model_path}' with training coverage: {best_coverage:.2f}% and holdout Accuracy: {best_acc}")
    else:
        print("No valid model found to save.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
