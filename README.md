# Learning Probability Density Function using GAN

## 📌 Project Overview

This project implements a **Generative Adversarial Network (GAN)** to learn the probability density function (PDF) of a transformed NO₂ concentration variable.

Instead of assuming a known analytical distribution, the GAN learns the distribution directly from real data samples.

Dataset: India Air Quality Dataset (Kaggle)  
Feature Used: NO₂ concentration  
Total Samples: 263,627 valid measurements  

---

## 🔢 Roll Number Parameterization

For roll number **102316130**, each NO₂ value `x` is transformed as:

z = x + aᵣ sin(bᵣ x)

Where:

- aᵣ = 0.5 × (r mod 7) = **1.0**
- bᵣ = 0.3 × ((r mod 5) + 1) = **0.3**

This introduces controlled non-linearity into the dataset.

The transformed data is normalized before training:

z_norm = (z − mean) / std

Normalization ensures stable GAN training.

---

## 🧠 GAN Architecture

### Generator
- Input: 1D Gaussian noise
- Layers: Linear(1→32) → ReLU → Linear(32→32) → ReLU → Linear(32→1)
- Output: Synthetic sample from learned distribution

### Discriminator
- Input: 1D real or fake sample
- Layers: Linear(1→32) → LeakyReLU → Linear(32→32) → LeakyReLU → Linear(32→1) → Sigmoid
- Output: Probability (real vs fake)

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Epochs | 4000 |
| Batch Size | 128 |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Loss Function | Binary Cross Entropy |
| Device | CPU/GPU |

Training alternates between:
- Updating Discriminator (real vs fake classification)
- Updating Generator (fooling the discriminator)

---

## 📊 PDF Approximation

After training:

- 10,000 samples are generated from the Generator
- Samples are denormalized
- Distribution comparison performed using:
  - Histogram overlay
  - Kernel Density Estimation (KDE)

---

## 📈 Results

### 1️⃣ Histogram Comparison
- Real transformed data vs GAN-generated samples
- Strong visual similarity
- Major distribution modes captured
![Histogram Comparison](images/histogram.png)


### 2️⃣ KDE-Based PDF Estimation
- Smooth continuous PDF curve
- Accurate approximation of empirical distribution
- No prior analytical form required
![KDE Curve](images/kde_curve.png)

---

## 🔍 Key Observations

- GAN successfully learned complex transformed NO₂ distribution
- Training remained stable across 4000 epochs
- Simple architecture (2 hidden layers) was sufficient
- Data normalization was critical for convergence
- Generated distribution closely matches empirical PDF

---

## ✅ Conclusion

This project demonstrates that **GANs can effectively learn probability density functions directly from sample data**, without assuming any known distribution.

The approach is especially useful when:

- The analytical PDF is unknown
- The distribution is complex or multimodal
- Traditional parametric modeling is difficult

GAN-based distribution learning provides a powerful generative modeling framework for real-world environmental datasets.

---
