# Learning Probability Density Function using GAN

## 📌 Project Overview

This project implements a **Generative Adversarial Network (GAN)** to learn the probability density function (PDF) of NO₂ (Nitrogen Dioxide) concentration data from the India Air Quality Dataset.

Instead of assuming any analytical distribution, the GAN learns the distribution directly from real data samples.

- **Dataset:** India Air Quality Dataset (Kaggle)
- **Feature Used:** NO₂ concentration
- **Total Samples:** 263,627 valid measurements
- **Framework:** PyTorch

---

## 🔢 Roll Number Parameterization

Roll Number: **102316130**

The transformation formula:

```
z = x + aᵣ sin(bᵣ x)
```

Where:

```
aᵣ = 0.5 × (r mod 7)
bᵣ = 0.3 × ((r mod 5) + 1)
```

For r = 102316130:

```
r mod 7 = 0
r mod 5 = 0
```

Therefore:

```
aᵣ = 0.0
bᵣ = 0.3
```

Since aᵣ = 0, the transformation reduces to:

```
z = x
```

So the GAN learns the **original NO₂ distribution**.

---

## 📊 Data Preprocessing

- Extract NO₂ column
- Remove missing values using `.dropna()`
- Convert to `float32`
- Normalize before training:

```
z_norm = (z - mean) / std
```

### Core Code (Data Processing)

```python
x = df["no2"].dropna().values.astype(np.float32)

r = 102316130
ar = 0.5 * (r % 7)
br = 0.3 * ((r % 5) + 1)

z = x + ar * np.sin(br * x)

z_mean, z_std = z.mean(), z.std()
z_norm = (z - z_mean) / z_std

z_tensor = torch.tensor(z_norm).view(-1,1).to(device)
```

---

## 🧠 GAN Architecture

### Generator

- Input: 1D Gaussian noise
- 2 hidden layers (32 neurons each)
- ReLU activation
- Output: 1D synthetic sample

```python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
```

---

### Discriminator

- Input: 1D sample
- 2 hidden layers (32 neurons each)
- LeakyReLU activation
- Sigmoid output

```python
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Epochs | 4000 |
| Batch Size | 128 |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Loss Function | Binary Cross Entropy |
| Device | CPU / GPU |

### Core Training Loop (Simplified)

```python
for epoch in range(epochs):

    # Train Discriminator
    idx = torch.randint(0, z_tensor.size(0), (batch_size,))
    real = z_tensor[idx]

    noise = torch.randn(batch_size,1).to(device)
    fake = G(noise).detach()

    d_loss = criterion(D(real), real_labels) + criterion(D(fake), fake_labels)

    d_opt.zero_grad()
    d_loss.backward()
    d_opt.step()

    # Train Generator
    noise = torch.randn(batch_size,1).to(device)
    fake = G(noise)

    g_loss = criterion(D(fake), real_labels)

    g_opt.zero_grad()
    g_loss.backward()
    g_opt.step()
```

---

## 📈 PDF Approximation

After training:

- Generate 10,000 samples
- Denormalize samples
- Compare distributions using:
  - Histogram
  - Kernel Density Estimation (KDE)

### Sample Generation

```python
G.eval()
with torch.no_grad():
    noise = torch.randn(10000,1).to(device)
    gen_z = G(noise).cpu().numpy()

gen_z = gen_z * z_std + z_mean
```

---

## 📊 Results

### 1️⃣ Histogram Comparison

Real vs GAN-generated distribution:

<p align="center">
  <img src="https://github.com/user-attachments/assets/25a0c775-4a11-4ad4-8df3-e1b8137bc88a" width="600">
</p>

---

### 2️⃣ KDE-Based PDF Estimation

Smooth estimated PDF from GAN samples:

<p align="center">
  <img src="https://github.com/user-attachments/assets/b2894b1f-883e-48f7-b459-f520f624cb95" width="600">
</p>

---

## 🔍 Key Observations

- GAN successfully learned the NO₂ distribution.
- Training remained stable over 4000 epochs.
- Even a simple architecture was sufficient for 1D modeling.
- Data normalization was critical for convergence.
- Generated distribution closely matches empirical PDF.

---

## ✅ Conclusion

This project demonstrates that **Generative Adversarial Networks can effectively learn probability density functions directly from sample data**, without requiring knowledge of the analytical distribution.

For roll number **102316130**, the transformation reduces to identity (z = x), and the GAN accurately models the original NO₂ concentration distribution.

GAN-based PDF learning is a powerful approach for modeling unknown or complex real-world distributions.

---
