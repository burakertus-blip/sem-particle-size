# Deep Learning Assisted Particle Size Ranking and Estimation from SEM Images

This repository contains the implementation code for the paper:

**"Deep Learning Assisted Particle Size Ranking and Estimation from 
SEM Images without Explicit Segmentation"**  
Emre Burak Ertuş — Micron (2026)  
DOI: https://doi.org/10.1016/j.micron.2026.104022

## Files
- `syntsem_v3.py` — Synthetic SEM dataset generation
- `ConvNext_v3.py` — Model architecture and training
- `predict_v3.py` — Inference and evaluation pipeline

## Requirements
Python 3.8+, PyTorch, OpenCV, NumPy, Pandas, scikit-learn, matplotlib

## Usage
1. Generate synthetic data: `python syntsem_v3.py`
2. Train the model: `python ConvNext_v3.py`
3. Run inference: `python predict_v3.py`
