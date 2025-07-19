Zero-Trust Tri-View IDS (ZTA)

This repository implements a **Zero-Trust anomaly detection system** with an adaptive learning pipeline combining:

- A Random Forest with tree replacement via **ACTER**
- Clustering-guided scoring with **K-Means Synergy**
- Structural drift signals using a **Leaf-Hash Neural Sleeve**
- Optional small language model (SLM) layer for pseudo-label refinement

Tested across two public datasets:
- **BCCC-CIC-Bell-DNS-2024** (offline training and evaluation)
- **UNSW-NB15** (cross-domain streaming validation)

---

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/bbkanomalyids/ZTA.git
cd ZTA

# Optional: use conda or venv
pip install -r requirements.txt
