# Conditional GAN (CGAN) on MNIST

![GANs Certificate](assets/gans_certificate.png)

*Completed DeepLearning.AI GAN Specialization (Oct 28, 2025)*

---

## 🎯 Overview

A **Conditional Generative Adversarial Network (CGAN)** implementation trained on the MNIST dataset to generate digit-specific images. This project serves as a practical application of the DeepLearning.AI GANs Specialization, demonstrating advanced generative modeling techniques.

### ✨ Key Features
- Conditional generation: Generate specific MNIST digits (0–9) on demand
- Training visualization: Monitor generator and discriminator loss dynamics
- Latent space exploration: Smooth interpolations between digit classes
- Comprehensive analysis: Detailed Jupyter notebook with training insights
- Modular, production-ready code structure for easy extension

---

## 📊 Results & Visualizations

### Generated Samples
| Digit 3 | Digit 7 | Digit 9 |
|---------|---------|---------|
| ![Digit 3](results/samples/custom_digit_3.png) | ![Digit 7](results/samples/custom_digit_7.png) | ![Digit 9](results/samples/custom_digit_9.png) |

### Latent Space Interpolation
**Smooth transition from 3 → 7:**
![Latent Interpolation](results/samples/interpolation.gif)

### Training Dynamics
**Discriminator vs. Generator Loss:**
![Training Loss](results/loss_curve.png)

---

## 🚀 Quick Start

### 1. Clone Repository

git clone https://github.com/yourusername/cgan-mnist.git
cd cgan-mnist


## 🛠️ Technology Stack

Deep Learning: PyTorch, torchvision

Data Handling & Visualization: pandas, numpy, matplotlib, seaborn

Development: Jupyter, ipykernel, tqdm

Image Processing: Pillow, OpenCV

## 🔬 Advanced Usage
Training Options
bash
Copy code
python -m src.train \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.0002 \
    --latent-dim 100 \
    --save-interval 5

## Generate multiple digits at once
python -m src.generate --digits 0 1 2 3 --count 16

## Create interpolation animation
python -m src.generate --interpolate 3 7 --steps 10 --gif

## Model Architecture
Generator: Fully connected network with conditional label embedding

Discriminator: Binary classifier with label conditioning

Conditioning: Label information injected into both networks

Optimization: Adam optimizer with tuned hyperparameters

## Learning Outcomes
GAN training dynamics and convergence challenges

Conditional generation techniques

Latent space manipulation and interpolation

Model evaluation and visualization strategies

Production-ready deep learning code structure

## Future Extensions
DCGAN architecture with convolutional layers

WGAN-GP for improved training stability

Class-conditional batch normalization

Real-time training visualization

Web interface for interactive generation
