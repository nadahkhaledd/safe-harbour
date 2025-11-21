# 🚀 Project Safe Harbour: Mars Landing Site Assessment

![Mars Banner](https://img.shields.io/badge/Mars-Safe%20Harbour-red) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

**Safe Harbour** is a Data Literacy Winter School project (2025) that utilizes Supervised Machine Learning to identify the safest landing zones on Mars. By fusing high-resolution terrain data (MOLA) with global climate simulations (GCM), we automate the risk assessment process for future robotic missions.

## 📖 Overview

Landing on Mars is notoriously difficult ("The Seven Minutes of Terror"). A successful site must satisfy two distinct sets of safety constraints:
1.  **The Ground:** Low elevation (for atmosphere), flat slopes, and low roughness.
2.  **The Sky:** High pressure, calm winds, low turbulence, and clear visibility.

This project moves beyond manual exclusion zones by training a **Random Forest Classifier** on 16 historical Mars missions (Ground Truth) to learn the complex, non-linear relationships between environmental parameters and mission success.

## 📂 Repository Structure

```text
safe-harbour/
├── data/
│   ├── gcm/              # Mars Climate Database (Yearly Stats & Raw Files)
│   ├── mola/             # MOLA Terrain Analysis (1° grid rank)
│   └── past_missions.csv # Ground Truth dataset (16 missions)
├── src/
│   ├── gcm_preprocessing.py           # Aggregates raw GCM data (min/max/mean)
│   ├── supervised_landing_analysis.py # Main ML pipeline (Training & Prediction)
│   ├── gcm_feature_selection.py       # Feature importance analysis script
│   └── plot_generation.py             # Visualization tools (2D Maps & 3D Globes)
├── output/
│   └── final/            # Generated maps, ROC curves, and confusion matrices
└── README.md             # This file
