# FakeImageScan  
**Beyond Visual Artifacts: Detecting AI-Generated Images via Model Affinity and Task Difficulty**

FakeImageScan is a **dual-pathway framework** for detecting AI-generated images produced by **GANs and diffusion models**.  
Instead of relying on fragile visual artifacts, FakeImageScan exploits **model affinity** and **task difficulty** signals to achieve **robust, generator-agnostic detection**.

---

## 📌 Key Idea

> **AI-generated images are more compatible with AI models than real images.**

FakeImageScan detects this asymmetry by jointly measuring:

- **How well AI models perform on an image** (*Model Affinity*), and  
- **How difficult the image should be for those tasks** (*Task Difficulty*).

A mismatch between these two signals provides a powerful cue for identifying synthetic images.

---

## 🔍 Framework Overview

![Framework_overview](Overview.PNG)

FakeImageScan consists of **two complementary pathways**:

### 1️⃣ Model Affinity Pathway (Performance Signals)

Measures how confidently AI models process an image.

- **Inpainting Accuracy (A)**  
  Evaluates reconstruction fidelity using **SSIM** after masked inpainting.  
  AI-generated images tend to be reconstructed with unusually high accuracy.

- **Segmentation Confidence (S)**  
  Measures average **per-pixel confidence** from a semantic segmentation model.  
  Synthetic images often produce over-confident and spatially uniform predictions.

---

### 2️⃣ Task Difficulty Pathway (Context Signals)

Estimates how challenging the image should be for vision models.

- **Pixel Naturalness (N)**  
  Quantifies pixel-level predictability using global and local probability models.

- **Image Complexity (C)**  
  Combines multiple structural and statistical cues, including:
  - Edge density  
  - Texture variance  
  - Entropy  
  - Frequency-domain characteristics  

---

### dataset structure
```plaintext
data/
├── Real/
│ ├── image1.png
│ ├── image2.png
│ ├── image3.png
│ └── ...
└── AI_generated/
├── image1.png
├── image2.png
├── image3.png
└── ...
```

---

## 🧠 Feature Vector

For each image **I**, FakeImageScan extracts a **4D feature vector**:

**x(I) = [ A(I), S(I), N(I), C(I) ]**

Where:
- **A(I)** — Inpainting Accuracy (SSIM-based)
- **S(I)** — Segmentation Confidence
- **N(I)** — Pixel Naturalness
- **C(I)** — Image Complexity

Final classification is performed using a **Support Vector Machine (SVM)** with an **RBF kernel**.


