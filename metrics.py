import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def compute_chunk_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute metrics for a single chunk.
    """
    metrics = {}
    metrics['accuracy_all'] = accuracy_score(y_true, y_pred)

    mask = (y_pred != -1)
    if mask.sum() > 0:
        metrics['accuracy_confident'] = accuracy_score(y_true[mask], y_pred[mask])
        metrics['recall_class_1'] = recall_score(y_true[mask], y_pred[mask], pos_label=1, zero_division=0)
        metrics['recall_class_0'] = recall_score(y_true[mask], y_pred[mask], pos_label=0, zero_division=0)
    else:
        metrics['accuracy_confident'] = np.nan
        metrics['recall_class_1'] = np.nan
        metrics['recall_class_0'] = np.nan

    metrics['precision_macro'] = precision_score(y_true, y_pred, labels=[0,1], average='macro', zero_division=0)
    metrics['f1_macro']      = f1_score(y_true, y_pred, labels=[0,1], average='macro', zero_division=0)
    metrics['uncertainty_rate'] = (y_pred == -1).mean()
    return metrics

def summarize_overall_metrics(all_y_true: list, all_y_pred: list) -> dict:
    """
    Compute overall metrics across all chunks and return with 'ALL' chunk_id.
    """
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    overall = compute_chunk_metrics(y_true, y_pred)
    overall.update({
        'chunk_id': 'ALL',
        'num_samples': len(y_true)
    })
    return overall
