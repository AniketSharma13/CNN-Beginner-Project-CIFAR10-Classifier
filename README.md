# 🧠 CNN Image Classifier — CIFAR-10

A beginner-level Deep Learning project that builds, trains, and evaluates a **Convolutional Neural Network (CNN)** from scratch using **PyTorch** on the **CIFAR-10** dataset.

---

## 📌 Project Overview

This project demonstrates the end-to-end workflow of a CNN-based image classification pipeline:

- Loading and preprocessing the CIFAR-10 dataset
- Designing a custom 3-layer CNN architecture
- Training the model with loss tracking
- Evaluating performance on a test set
- Visualizing training loss and validation accuracy curves
- Saving the trained model weights

**Final Test Accuracy:** ~`XX%` *(update after running)*

---

## 🗂️ Repository Structure

```
CNN-CIFAR10-Classifier/
│
├── CNN_for_Cipher.ipynb     # Main Jupyter notebook (training + evaluation)
├── Cnn_Model.pth            # Saved model weights (PyTorch state_dict)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── assets/                  # (Optional) Sample plots/images
    ├── loss_curve.png
    └── accuracy_curve.png
```

---

## 🏗️ Model Architecture

The CNN consists of three convolutional blocks followed by two fully connected layers:

```
Input: (3, 32, 32) — RGB image

Conv Block 1:  Conv2d(3 → 32,  kernel=3, padding=1) → ReLU → MaxPool(2x2)
Conv Block 2:  Conv2d(32 → 64, kernel=3, padding=1) → ReLU → MaxPool(2x2)
Conv Block 3:  Conv2d(64 → 128,kernel=3, padding=1) → ReLU → MaxPool(2x2)

Flatten: 128 × 4 × 4 = 2048

FC Layer 1:    Linear(2048 → 256) → ReLU
FC Layer 2:    Linear(256 → 10)   → Output (10 classes)
```

| Layer | Type | Output Shape |
|---|---|---|
| Input | — | (3, 32, 32) |
| Conv1 + Pool | Conv2d + MaxPool | (32, 16, 16) |
| Conv2 + Pool | Conv2d + MaxPool | (64, 8, 8) |
| Conv3 + Pool | Conv2d + MaxPool | (128, 4, 4) |
| Flatten | — | (2048,) |
| FC1 | Linear + ReLU | (256,) |
| FC2 | Linear | (10,) |

---

## 📦 Dataset — CIFAR-10

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is a standard benchmark dataset containing **60,000 color images** (32×32 pixels) across **10 classes**:

| Label | Class |
|---|---|
| 0 | ✈️ Airplane |
| 1 | 🚗 Automobile |
| 2 | 🐦 Bird |
| 3 | 🐱 Cat |
| 4 | 🦌 Deer |
| 5 | 🐶 Dog |
| 6 | 🐸 Frog |
| 7 | 🐴 Horse |
| 8 | 🚢 Ship |
| 9 | 🚚 Truck |

- **Training set:** 50,000 images
- **Test set:** 10,000 images

The dataset is automatically downloaded via `torchvision.datasets.CIFAR10`.

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (default lr=0.001) |
| Loss Function | CrossEntropyLoss |
| Batch Size | 64 |
| Epochs | 10 |
| Normalization | Mean=0.5, Std=0.5 (per channel) |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/CNN-CIFAR10-Classifier.git
cd CNN-CIFAR10-Classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

Open the notebook in Jupyter or VS Code:

```bash
jupyter notebook CNN_for_Cipher.ipynb
```

Run all cells top to bottom. The CIFAR-10 dataset will be downloaded automatically on first run into a `./data/` folder.

### 4. Load the Pretrained Model (Optional)

To use the saved weights without retraining:

```python
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convo_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(4 * 4 * 128, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.convo_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

model = CNN()
model.load_state_dict(torch.load("Cnn_Model.pth", map_location=torch.device("cpu")))
model.eval()
print("Model loaded successfully!")
```

---

## 📊 Results

*(Update these after running the notebook)*

| Metric | Value |
|---|---|
| Final Training Loss | ~X.XX |
| Final Validation Loss | ~X.XX |
| Test Accuracy | ~XX% |

Training loss and accuracy plots are generated at the end of the notebook.

---

## 🧰 Tech Stack

- **Python** 3.8+
- **PyTorch** — Model building, training, and saving
- **TorchVision** — Dataset loading and image transforms
- **Matplotlib** — Visualization of loss and accuracy curves

---

## 📋 Requirements

See `requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
jupyter>=1.0.0
```

---

## 🔮 Future Improvements

- [ ] Add Batch Normalization after each conv layer
- [ ] Experiment with Dropout for regularization
- [ ] Try deeper architectures (ResNet-style skip connections)
- [ ] Implement data augmentation (random flips, crops)
- [ ] Train for more epochs with a learning rate scheduler
- [ ] Add per-class accuracy breakdown
- [ ] Export model to ONNX format for deployment

---

## 🙋 Author

**[Your Name]**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

> 💡 *Built as a beginner deep learning project to understand CNN fundamentals using PyTorch and the CIFAR-10 benchmark.*
