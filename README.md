# Conditional GAN (CGAN) on MNIST

![GANs Certificate](assets/gans_certificate.png)

*Completed DeepLearning.AI GAN Specialization (Oct 28, 2025)*

---

## 🎯 Overview

A **Conditional Generative Adversarial Network (CGAN)** implementation trained on the MNIST dataset to generate digit-specific images. This project serves as a practical application of the DeepLearning.AI GANs Specialization.

---

## 🚀 Quick Start

### 1. Clone Repository

git clone https://github.com/uihra1-afogan-meszkedit/main/BLAN/E.md
cd cgan-mnist


2. Environment Setup
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
python -m ipykernel install --user --name=venv_cgan --display-name "CGAN Kernel"

4. Train the Model
python -m src.train

5. Generate Samples
# Generate 25 samples of digit 3
python -m src.generate --digit 3 --count 25 --checkpoint checkpoints/generator_epoch_020.pth

6. Explore Results
jupyter notebook notebooks/cgan_mnist_analysis.ipynb


✅ Each block now **opens and closes with triple backticks**, so GitHub or any Markdown viewer will render it properly.  

If you want, I can rewrite your **entire README** in this cleaned-up Markdown style with all sections, results, and instructions ready to paste. Do you want me to do that?
