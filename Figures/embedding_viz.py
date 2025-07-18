#!/usr/bin/env python3
# embedding_viz.py
#
# Generate a 2D projection (t-SNE or UMAP) of your pipeline’s leaf embeddings,
# colored by ground-truth label, pseudo-label, and chunk index (time).

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Optional UMAP
try:
    import umap
except ImportError:
    umap = None

# Pipeline modules
from config import (
    ONLINE_DATA_PATH,
    KMEANS_PATH,
    RF_PATH,
    LEAF_DIM,
    FINAL_FEATURES,
    KMEANS_FEATURES,
    CHUNK_SIZE,
    CONFIDENCE_THRESH,
)
from data import load_online_data
from features import compute_kmeans_synergy
from models import load_models, get_prediction_with_confidence
from leaf_view import extract_leaf_embed


def parse_args():
    parser = argparse.ArgumentParser(
        description="2D projection of leaf-embedding vectors (t-SNE or UMAP)."
    )
    parser.add_argument(
        "--method", choices=["tsne", "umap"], default="tsne",
        help="Dimensionality reduction: t-SNE or UMAP"
    )
    parser.add_argument(
        "--perplexity", type=float, default=30.0,
        help="Perplexity (t-SNE) or n_neighbors (UMAP)"
    )
    parser.add_argument(
        "--n-iter", type=int, default=1000,
        help="Max iterations (t-SNE) or n_epochs (UMAP)"
    )
    parser.add_argument(
        "--scale", action="store_true",
        help="Standard-scale embeddings before projection"
    )
    parser.add_argument(
        "--output", type=str, default="leaf_embedding_2d.png",
        help="Path to save the output PNG"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load & preprocess full online dataset (includes 'label')
    df = load_online_data(
        path=str(ONLINE_DATA_PATH),
        final_features=FINAL_FEATURES,
        drop_label=False
    )

    # 2. Add synergy features via KMeans
    kmeans, rf_model = load_models(
        str(KMEANS_PATH), str(RF_PATH), FINAL_FEATURES
    )
    df = compute_kmeans_synergy(df, kmeans, KMEANS_FEATURES)

    # 3. Extract feature matrix for embedding and prediction
    X = df[FINAL_FEATURES].to_numpy(dtype=float)

    # 4. Compute leaf-embedding vectors (Z: n_samples x LEAF_DIM)
    Z = extract_leaf_embed(X, rf_model, LEAF_DIM)

    # 5. Labels: ground truth and pseudo-labels
    y_true = df['label'].to_numpy()
    y_pred, _ = get_prediction_with_confidence(rf_model, X, CONFIDENCE_THRESH)

    # 6. Chunk indices (1-based)
    idx = np.arange(len(df))
    chunk_idx = (idx // CHUNK_SIZE) + 1

    # 7. Optional scaling and PCA pre-reduction for speed
    if args.scale:
        Z = StandardScaler().fit_transform(Z)
    if Z.shape[1] > 50:
        Z = PCA(n_components=50, random_state=42).fit_transform(Z)

    # 8. Dimensionality reduction to 2D
    if args.method == 'tsne':
        reducer = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            n_iter=args.n_iter,
            random_state=42
        )
        Z2 = reducer.fit_transform(Z)
    else:
        if umap is None:
            print(
                "ERROR: umap-learn is not installed. Install with `pip install umap-learn`.",
                file=sys.stderr
            )
            sys.exit(1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(args.perplexity),
            n_epochs=args.n_iter,
            random_state=42
        )
        Z2 = reducer.fit_transform(Z)

    # 9. Plot up to three panels
    n_panels = 3
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(6 * n_panels, 6),
        squeeze=False
    )

    def scatter(ax, c, title, label):
        sc = ax.scatter(
            Z2[:, 0], Z2[:, 1],
            c=c,
            cmap='coolwarm',
            s=10,
            alpha=0.7,
            edgecolors='none'
        )
        ax.set_title(title)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(label)

    scatter(axes[0, 0], y_true,    'Projection: True Label',   'True Label')
    scatter(axes[0, 1], y_pred,    'Projection: Pseudo-Label', 'Pseudo Label')
    scatter(axes[0, 2], chunk_idx, 'Projection: Chunk Index',  'Chunk Index')

    plt.tight_layout()
    fig.savefig(args.output, dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
