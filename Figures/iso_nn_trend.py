#!/usr/bin/env python3
# iso_nn_trend.py
"""
Plot ISO/RF, NN/RF, and RF tree-agreement trends over chunks with markers for ACTER triggers.

Usage:
    python iso_nn_trend.py --diag acter_diag.json --output iso_nn_trend.png
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot isolation forest, NN, and RF agreement trends over chunks."  
    )
    parser.add_argument(
        "--diag",
        type=Path,
        default=Path("acter_diag.json"),
        help="Path to acter_diag.json file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("iso_nn_trend.png"),
        help="Path to save the trend plot PNG",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load diagnostics JSON
    if not args.diag.exists():
        print(f"ERROR: {args.diag} not found", file=sys.stderr)
        sys.exit(1)
    df = pd.read_json(args.diag)

    # Ensure required columns exist
    required = {'iso_agree', 'nn_agree', 'mean_agree', 'triggered', 'chunk'}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: JSON missing fields: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # Sort by chunk index
    df = df.sort_values(by='chunk')
    chunks = df['chunk']
    iso = df['iso_agree']
    nn  = df['nn_agree']
    rf  = df['mean_agree']
    triggered = df['triggered'].astype(bool)

    # Plot trends
    plt.figure(figsize=(10, 5))
    plt.plot(chunks, iso, marker='o', linestyle='-', label='ISO/RF Agreement')
    plt.plot(chunks, nn,  marker='s', linestyle='-', label='NN/RF Agreement')
    plt.plot(chunks, rf,  marker='^', linestyle='-', label='RF Tree-Agreement')

    # Overlay circles at triggered chunks
    xs = chunks[triggered]
    plt.scatter(xs, iso[triggered], facecolors='none', edgecolors='black', s=150)
    plt.scatter(xs, nn[triggered],  facecolors='none', edgecolors='black', s=150)
    plt.scatter(xs, rf[triggered],  facecolors='none', edgecolors='black', s=150)

    plt.title('Agreement Trends Over Chunks (ACTER Triggers Marked)')
    plt.xlabel('Chunk Index')
    plt.ylabel('Agreement Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and show
    plt.savefig(args.output, dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
