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

## 📁 Repository Structure

cgan-mnist/
├── src/ # Source code
│ ├── train.py # Training script
│ ├── generate.py # Generation utilities
│ ├── utils.py # Utility functions
│ └── models/
│ ├── generator.py # CGAN generator architecture
│ └── discriminator.py # CGAN discriminator architecture
├── notebooks/
│ └── cgan_mnist_analysis.ipynb # Complete analysis notebook
├── checkpoints/ # Saved model weights
├── results/
│ ├── samples/ # Generated images
│ └── training/ # Loss curves & metrics
├── assets/ # Documentation assets
├── requirements.txt # Python dependencies
└── README.md # Project overview

yaml
Copy code

---

## 📊 Results & Visualizations

### Generated Samples
| Digit 3 | Digit 7 | Digit 9 |
|---------|---------|---------|
| ![Digit 3](results/samples/digit_3_grid.png) | ![Digit 7](results/samples/digit_7_grid.png) | ![Digit 9](results/samples/digit_9_grid.png) |

### Latent Space Interpolation
**Smooth transition from 3 → 7:**
![Latent Interpolation](results/samples/latent_interpolation_3_7.png)

### Training Dynamics
**Discriminator vs. Generator Loss:**
![Training Loss](results/training/loss_curves.png)

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/cgan-mnist.git
cd cgan-mnist
2. Environment Setup
bash
Copy code
# Create virtual environment
python -m venv venv_cgan

# Activate environment
# Linux/macOS
source venv_cgan/bin/activate
# Windows
venv_cgan\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Jupyter Integration (Optional)
bash
Copy code
python -m ipykernel install --user --name=venv_cgan --display-name "CGAN Kernel"
4. Train the Model
bash
Copy code
python -m src.train
5. Generate Samples
bash
Copy code
# Generate 25 samples of digit 3
python -m src.generate --digit 3 --count 25 --checkpoint checkpoints/generator_epoch_020.pth
6. Explore Results
bash
Copy code
jupyter notebook notebooks/cgan_mnist_analysis.ipynb
🛠️ Technology Stack
Deep Learning: PyTorch, torchvision

Data Handling & Visualization: pandas, numpy, matplotlib, seaborn

Development: Jupyter, ipykernel, tqdm

Image Processing: Pillow, OpenCV

🔬 Advanced Usage
Training Options
bash
Copy code
python -m src.train \
    --epochs 50 \
    --batch-size 64 \
    --learning-rate 0.0002 \
    --latent-dim 100 \
    --save-interval 5
Generation Features
bash
Copy code
# Generate multiple digits at once
python -m src.generate --digits 0 1 2 3 --count 16

# Create interpolation animation
python -m src.generate --interpolate 3 7 --steps 10 --gif
🏗️ Model Architecture
Generator: Fully connected network with conditional label embedding

Discriminator: Binary classifier with label conditioning

Conditioning: Label information injected into both networks

Optimization: Adam optimizer with tuned hyperparameters

🎓 Learning Outcomes
GAN training dynamics and convergence challenges

Conditional generation techniques

Latent space manipulation and interpolation

Model evaluation and visualization strategies

Production-ready deep learning code structure

🔮 Future Extensions
DCGAN architecture with convolutional layers

WGAN-GP for improved training stability

Class-conditional batch normalization

Real-time training visualization

Web interface for interactive generation
