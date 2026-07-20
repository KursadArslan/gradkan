# GradKAN: Enhanced Kolmogorov-Arnold Networks with Adaptive Knot Allocation

This repository contains the official PyTorch implementation of **GradKAN** (*Gradient-Adaptive Kolmogorov-Arnold Networks*).

GradKAN replaces standard static/uniform knot allocation with a dynamic, gradient-aware grid update mechanism. By leveraging Exponential Moving Average (EMA) filtering and PDF/CDF transformations, GradKAN shifts B-spline knots into high-gradient regions without increasing the total parameter budget.

---

## 🚀 Features
* **Gradient-Aware Allocation**: Dynamically shifts B-spline knots to high-error regions.
* **Training Stability**: Uses Exponential Moving Average (EMA) to prevent grid oscillation.
* **Parameter Efficient**: Outperforms baseline KAN with zero added parameters.
* **Weight Re-projection**: Preserves historical learning using PyTorch Least Squares (`torch.linalg.lstsq`).

---

## 📦 Installation

Clone the repository and install dependencies:
