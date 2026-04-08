# XAI-BraTS: Explainable & Generative AI for Brain Tumor MRI Analysis

## Description

A research-focused framework for Brain Tumor Segmentation that bridges the gap between high-performance Deep Learning and clinical trust. This repository implements Generative AI for high-fidelity MRI synthesis and Explainable AI (XAI) techniques to provide transparent, uncertainty-aware clinical decision support.

## Key Features (The "Core" of your Repo)

- 🧠 Spatio-Temporal Segmentation: Multi-sequence MRI analysis (T1, T1ce, T2, FLAIR).
- 🎨 Generative Data Augmentation: Using Diffusion or GAN-based models to handle class imbalance in tumor sub-regions.
- 🔍 Explainable AI (XAI): Integrated saliency mapping and attention mechanisms to visualize model focus.
- 🎲 Uncertainty Quantification: Bayesian or Monte Carlo Dropout layers to provide "confidence scores" for every prediction.
- 📊 Evaluation: Rigorous benchmarking on the BraTS dataset using Dice, Hausdorff Distance, and ECE (Expected Calibration Error).