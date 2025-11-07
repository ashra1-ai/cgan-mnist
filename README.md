<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CGAN MNIST Project</title>
    <style>
        body {
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
            color: #333;
        }
        .container {
            max-width: 900px;
            margin: auto;
            padding: 2rem;
            background-color: #fff;
        }
        h1, h2, h3 {
            color: #222;
        }
        h1 {
            border-bottom: 2px solid #444;
            padding-bottom: 0.5rem;
        }
        img {
            max-width: 100%;
            height: auto;
        }
        code, pre {
            background-color: #f0f0f0;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
        }
        pre {
            padding: 1rem;
            overflow-x: auto;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1rem 0;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 0.75rem;
            text-align: center;
        }
        th {
            background-color: #eee;
        }
        a {
            color: #0066cc;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .section {
            margin-top: 2rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Conditional GAN (CGAN) on MNIST</h1>

        <img src="assets/gans_certificate.png" alt="GANs Certificate">
        <p><em>Completed DeepLearning.AI GAN Specialization (Oct 28, 2025)</em></p>

        <div class="section">
            <h2>Overview</h2>
            <p>This repository demonstrates a <strong>Conditional Generative Adversarial Network (CGAN)</strong> trained on the MNIST dataset. The project was developed as a <strong>practical implementation</strong> for the DeepLearning.AI <a href="https://www.deeplearning.ai/courses/generative-adversarial-networks-gans/">Generative Adversarial Networks (GANs) Specialization</a>.</p>

            <ul>
                <li>Digit-specific image generation using CGAN.</li>
                <li>Visualization of <strong>generator</strong> and <strong>discriminator losses</strong> during training.</li>
                <li>Latent space interpolation between digits.</li>
                <li>Fully documented <strong>Jupyter notebook analysis</strong> of generated images and training metrics.</li>
            </ul>

            <p>This work reinforces my understanding of GAN architectures, adversarial training, and generative modeling — skills directly applicable to my upcoming research on <strong>EGEAT (Exact Geometric Ensemble Adversarial Training)</strong> for robust deep learning.</p>
        </div>

        <div class="section">
            <h2>Repository Structure</h2>
            <pre>
cgan-mnist/
├── README.md
├── LICENSE
├── requirements.txt
├── config.py
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── generate.py
│   ├── utils.py
│   └── models/
│       ├── __init__.py
│       ├── generator.py
│       └── discriminator.py
├── checkpoints/
│   └── generator_epoch_020.pth
├── results/
│   ├── samples/
│   │   ├── custom_digit_3.png
│   │   ├── custom_digit_7.png
│   │   └── latent_interp_3_7.png
│   └── training_log.csv
├── notebooks/
│   └── cgan_mnist_analysis.ipynb
└── assets/
    └── gans_certificate.png
            </pre>
        </div>

        <div class="section">
            <h2>Results</h2>
            <h3>Generated Samples by Digit</h3>
            <table>
                <tr>
                    <th>Digit 3</th>
                    <th>Digit 7</th>
                </tr>
                <tr>
                    <td><img src="results/samples/custom_digit_3.png" alt="Digit 3"></td>
                    <td><img src="results/samples/custom_digit_7.png" alt="Digit 7"></td>
                </tr>
            </table>

            <h3>Latent Space Interpolation</h3>
            <p>Interpolating between 3 → 7:</p>
            <img src="results/samples/latent_interp_3_7.png" alt="Latent Interpolation">

            <h3>Training Loss Curves</h3>
            <p>Discriminator vs. Generator losses:</p>
            <img src="results/samples/loss_plot.png" alt="Training Loss">
        </div>

        <div class="section">
            <h2>Quick Start</h2>
            <ol>
                <li>
                    <strong>Clone the repository:</strong>
                    <pre>git clone https://github.com/yourusername/cgan-mnist.git
cd cgan-mnist</pre>
                </li>
                <li>
                    <strong>Create a virtual environment:</strong>
                    <pre>python -m venv venv_cgan
# Linux/macOS
source venv_cgan/bin/activate
# Windows
venv_cgan\Scripts\activate</pre>
                </li>
                <li>
                    <strong>Install dependencies:</strong>
                    <pre>pip install -r requirements.txt</pre>
                </li>
                <li>
                    <strong>Optional: Add virtual environment to Jupyter:</strong>
                    <pre>python -m ipykernel install --user --name=venv_cgan --display-name "CGAN venv"</pre>
                </li>
                <li>
                    <strong>Train the CGAN (optional):</strong>
                    <pre>python -m src.train</pre>
                </li>
                <li>
                    <strong>Generate digit-specific samples:</strong>
                    <pre>python -m src.generate --digit 3 --count 25 --checkpoint checkpoints/generator_epoch_020.pth</pre>
                </li>
                <li>
                    <strong>Explore results in the notebook:</strong>
                    <pre>jupyter notebook notebooks/cgan_mnist_analysis.ipynb</pre>
                </li>
            </ol>
        </div>

        <div class="section">
            <h2>Technologies & Tools</h2>
            <ul>
                <li>Deep Learning: PyTorch, torchvision</li>
                <li>Data Handling & Visualization: pandas, numpy, matplotlib, seaborn, tqdm</li>
                <li>Notebook Support: Jupyter, ipykernel</li>
                <li>Image Utilities: Pillow</li>
            </ul>
        </div>

        <div class="section">
            <h2>Analysis Notebook</h2>
            <p>The notebook <code>cgan_mnist_analysis.ipynb</code> provides:</p>
            <ul>
                <li>Digit-specific sample generation using the trained generator.</li>
                <li>Visualization of generator and discriminator losses from <code>training_log.csv</code>.</li>
                <li>Latent space interpolation between any two digits.</li>
                <li>Optional exploration of generator behavior for custom noise vectors and labels.</li>
            </ul>
        </div>

        <div class="section">
            <h2>License</h2>
            <p>This project is licensed under the MIT License. See <a href="LICENSE">LICENSE</a> for details.</p>
        </div>

        <div class="section">
            <h2>References</h2>
            <ol>
                <li>DeepLearning.AI. <a href="https://www.deeplearning.ai/courses/generative-adversarial-networks-gans/">Generative Adversarial Networks Specialization</a></li>
                <li>Goodfellow, I. et al., <em>Generative Adversarial Networks</em>, 2014</li>
                <li>EGEAT Research (upcoming): Exact Geometric Ensemble Adversarial Training</li>
            </ol>
        </div>

        <div class="section">
            <h2>Portfolio & Research Notes</h2>
            <ul>
                <li>Demonstrates practical mastery of CGANs, from training to visualization.</li>
                <li>Generated samples and interpolations can be directly included in presentations or research papers.</li>
                <li>Notebook-based analysis makes it easy to reproduce results and validate model behavior.</li>
                <li>Reinforces knowledge for robust adversarial training, generative modeling, and PyTorch workflows.</li>
                <li>Certificate included in <code>assets/</code> reinforces credibility and skill completion.</li>
            </ul>
        </div>
    </div>
</body>
</html>
