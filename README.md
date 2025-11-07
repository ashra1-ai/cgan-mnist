# Conditional GAN (CGAN) on MNIST

![GANs Certificate](assets/gans_certificate.png)

*Completed DeepLearning.AI GAN Specialization (Oct 28, 2025)*

---

## Overview

This repository demonstrates a **Conditional Generative Adversarial Network (CGAN)** trained on the MNIST dataset. This project was developed as a **practical implementation** for the DeepLearning.AI [Generative Adversarial Networks (GANs) Specialization](https://www.deeplearning.ai/courses/generative-adversarial-networks-gans/).

**Key Features:**

- Digit-specific image generation using CGAN.
- Visualization of **generator** and **discriminator losses** during training.
- Latent space interpolation between digits.
- Fully documented **Jupyter notebook analysis** of generated images and training metrics.

This work reinforces my understanding of GAN architectures, adversarial training, and generative modeling — skills directly applicable to my upcoming research on **EGEAT (Exact Geometric Ensemble Adversarial Training)** for robust deep learning.

---

## Repository Structure


---

## 📈 Results

**Generated Samples by Digit**

| Digit 3 | Digit 7 |
|---------|---------|
| ![Digit 3](results/samples/custom_digit_3.png) | ![Digit 7](results/samples/custom_digit_7.png) |

**Latent Space Interpolation**

Interpolating between 3 → 7:

![Latent Interpolation](results/samples/latent_interp_3_7.png)

**Training Loss Curves**

Discriminator vs. Generator losses:

![Training Loss](results/samples/loss_plot.png)

---

# ⚡ Quick Start Guide
quick_start:
  - step: "Clone the repository"
    command: |
      git clone https://github.com/yourusername/cgan-mnist.git
      cd cgan-mnist

  - step: "Create a virtual environment"
    commands:
      linux_macos: |
        python -m venv venv_cgan
        source venv_cgan/bin/activate
      windows: |
        python -m venv venv_cgan
        venv_cgan\Scripts\activate

  - step: "Install dependencies"
    command: "pip install -r requirements.txt"

  - step: "Optional: Add virtual environment to Jupyter"
    command: "python -m ipykernel install --user --name=venv_cgan --display-name \"CGAN venv\""

  - step: "Train the CGAN (optional)"
    command: "python -m src.train"

  - step: "Generate digit-specific samples"
    command: "python -m src.generate --digit 3 --count 25 --checkpoint checkpoints/generator_epoch_020.pth"

  - step: "Explore results in the notebook"
    command: "jupyter notebook notebooks/cgan_mnist_analysis.ipynb"

# 🛠 Technologies & Tools
technologies:
  deep_learning:
    - "PyTorch"
    - "torchvision"
  
  data_handling_visualization:
    - "pandas"
    - "numpy" 
    - "matplotlib"
    - "seaborn"
    - "tqdm"

  notebook_support:
    - "Jupyter"
    - "ipykernel"

  image_utilities:
    - "Pillow"

# Project Structure
project_structure:
  - "src/"
    - "train.py"
    - "generate.py"
  - "notebooks/"
    - "cgan_mnist_analysis.ipynb"
  - "checkpoints/"
  - "requirements.txt"
