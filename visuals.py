import matplotlib.pyplot as plt
import numpy as np


def plot_pseudo_label_distribution(counts: list, output_path: str = None) -> None:
    """
    Plot pseudo-label distribution across chunks.

    Args:
        counts: list of dicts mapping label to count per chunk.
        output_path: if provided, path to save the figure; otherwise show interactively.
    """
    indices = range(1, len(counts) + 1)
    ones      = [c.get(1,   0) for c in counts]
    zeros     = [c.get(0,   0) for c in counts]
    uncertain = [c.get(-1,  0) for c in counts]

    plt.figure(figsize=(10, 5))
    plt.plot(indices, ones,      marker='o', label='Anomalies (1)')
    plt.plot(indices, zeros,     marker='x', label='Normal (0)')
    plt.plot(indices, uncertain, marker='s', label='Uncertain (-1)')
    plt.title('Pseudo-Label Distribution Over Chunks')
    plt.xlabel('Chunk Index')
    plt.ylabel('Sample Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_academic_panels(counts: list, diag: list, output_path: str = None) -> None:
    """
    Create a 3-row figure for research-grade panels:
      1) Stacked pseudo-label volumes with ACTER triggers.
      2) Mean agreement ±1σ ribbon to highlight drift spikes.
      3) Tri-confident pool growth over time.

    Args:
        counts: list of dicts mapping label->count per chunk.
        diag:   list of dicts per chunk with keys:
                'triggered' (bool), 'mean_agree' (float),
                'std_agree' (float), 'tri_conf' (int).
        output_path: if provided, save figure there; otherwise show.
    """
    n = len(counts)
    x = np.arange(1, n + 1)

    # Optional publication style for saved figures
    if output_path:
        plt.style.use("seaborn-v0_8-paper")

    # Unpack and sanitize data
    normal    = np.array([float(c.get(0, 0))   for c in counts])
    anomaly   = np.array([float(c.get(1, 0))   for c in counts])
    uncertain = np.array([float(c.get(-1, 0))  for c in counts])

    triggered = np.array([bool(d.get('triggered', False)) for d in diag])
    mean_ag   = np.array([float(d.get('mean_agree',   0.0)) for d in diag])
    std_ag    = np.nan_to_num(
        np.array([d.get('std_agree', 0.0) for d in diag], dtype=float),
        nan=0.0
    )
    tri_conf  = np.array([int(d.get('tri_conf',   0))   for d in diag])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # ── Panel 1: stacked volumes + ACTER triggers ────────────────────────
    axes[0].stackplot(
        x,
        normal, anomaly, uncertain,
        labels=['Normal', 'Anomaly', 'Uncertain'],
        colors=['#4CAF50', '#F44336', '#FFC107']
    )
    # vertical lines where adaptation was triggered
    for idx in np.where(triggered)[0]:
        axes[0].axvline(x=idx + 1, color='red', linestyle='--', linewidth=1)
    axes[0].legend(loc='upper right')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Pseudo-Label Volumes & ACTER Events')

    # ── Panel 2: agreement mean ±1σ ─────────────────────────────────────
    axes[1].plot(x, mean_ag, label='Mean Agreement', linewidth=2)
    axes[1].fill_between(
        x,
        mean_ag - std_ag,
        mean_ag + std_ag,
        alpha=0.3,
        label='±1 σ'
    )
    axes[1].set_ylabel('Agreement')
    axes[1].set_title('Forest Agreement (Mean ±1σ)')
    axes[1].legend(loc='upper right')

    # ── Panel 3: tri-confident pool growth ──────────────────────────────
    axes[2].bar(x, tri_conf, width=0.8)
    axes[2].set_xlabel('Chunk Index')
    axes[2].set_ylabel('Tri-Confident Size')
    axes[2].set_title('Tri-Confident Pool Growth')

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
    else:
        plt.show()
    plt.close()
