# Visual Privacy Data Analysis

This repository contains the code and analysis pipeline for studying how spatial configuration shapes **perceived visual privacy** in dense housing using BIM-generated scenes, screen renders, and immersive VR. The workflow implements a **leakage-safe, uncertainty-aware, and interpretable** machine learning pipeline, with a focus on **transfer learning from render data to VR**.

## Overview
From **160 parametric BIM scenes**, we collected **3,587 ratings** from screen renders and **450 ratings** from immersive VR. The code here:

- Applies **group-aware, template-disjoint splits** to avoid leakage between similar scenes.
- Trains tree-ensemble models (e.g., Gradient Boosting, XGBoost) for **explanatory analysis, not deployment-level prediction**.
- Uses **transfer learning (Render+VR → VR)** to improve immersive models under limited VR data.
- Provides **interpretable explanations** via SHAP and Accumulated Local Effects (ALE).
- Calibrates **uncertainty** using ICC-like noise ceilings, **bootstrap intervals** for test \(R^2\), and **split-conformal prediction intervals**.
- Includes **drop-group ablations** and **residual analysis** to understand which feature families and geometries matter most.

## Key components

The analysis pipeline is organized conceptually into the following steps:

1. **Data preparation**  
   - Load and clean BIM-derived features and survey ratings (Render + VR).  
   - Define scene “templates” and create **template-disjoint** train/validation/test splits.

2. **Model training and selection**  
   - Train baseline and tree-ensemble models under group-aware evaluation.  
   - Select domain-best models for Render, VR, and Combined (Render+VR).

3. **Noise ceilings and normalized performance**  
   - Estimate **ICC-like reliability ceilings** for each domain.  
   - Compare test \(R^2\) to ceilings to compute the fraction of explainable variance captured.

4. **Transfer learning and cross-modality tests**  
   - Implement **Render-only**, **VR-only**, and **Render+VR → VR** transfer configurations.  
   - Evaluate cross-modality generalization (train in one domain, test in another).

5. **Interpretability and design levers**  
   - Compute **SHAP values** (including interactions), with permutation-based BH–FDR significance and rank stability checks.  
   - Use **ALE** near the median to obtain “actionable deltas” for selected features.  
   - Run **drop-group ablations** (Views, Context, Openings, Treatments, Balconies, Demographics) to quantify necessity.

6. **Uncertainty and error analysis**  
   - Estimate **bootstrap confidence intervals** for test \(R^2\).  
   - Compute **split-conformal 90% prediction intervals** for per-scene scores.  
   - Inspect residuals to identify error hotspots and data gaps (e.g., rare tight setbacks or heavily screened façades).

## Data

The raw survey data are **not** included in this repository (e.g., due to size or privacy constraints).  
The code assumes preprocessed tabular data with:

- **Input features** such as sill/sight heights, window widths, building/street geometry, balconies, curtains, and basic demographics (age, sex, education).  
- **Target variables**: mean perceived privacy ratings for Render and/or VR scenes on a 1–5 scale.  

You can adapt the data-loading scripts to your own schema by updating the relevant file paths and column names.

## Getting started

### Requirements

numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
shap
scipy
ipython  # optional, used for display helpers in notebooks

Once you have cloned the repository, install dependencies with:

```bash
pip install -r requirements.txt
