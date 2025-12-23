# FakeImageScan  
**Beyond Visual Artifacts: Detecting AI-Generated Images via Model Affinity and Task Difficulty**

FakeImageScan is a **dual-pathway framework** for detecting AI-generated images produced by **GANs and diffusion models**.  
Instead of relying on fragile visual artifacts or generator-specific fingerprints, FakeImageScan exploits **model affinity** and **task difficulty** signals to achieve **robust, generator-agnostic detection**.

---

## 📌 Key Insight

> **AI-generated images are intrinsically more compatible with AI vision models than real images.**

FakeImageScan detects this asymmetry by jointly measuring:
- **How well AI models perform on an image** (*Model Affinity*), and  
- **How difficult the image should be for those tasks** (*Task Difficulty*).

A mismatch between these two signals provides a powerful cue for identifying synthetic images.

---

## 🔍 Framework Overview

![Framework Overview](Overview.PNG)

FakeImageScan consists of **two complementary pathways** that operate in parallel on each input image.

---

## 📁 Dataset Structure

The dataset follows a **binary folder structure** with two classes: **Real** and **AI-Generated** images.

```plaintext
data/
├── Real/
│   ├── image1.png
│   ├── image2.png
│   ├── image3.png
│   └── ...
└── AI_generated/
    ├── image1.png
    ├── image2.png
    ├── image3.png
    └── ...
```
### 🔍 Feature Semantics

The four-dimensional feature vector is derived from two complementary pathways that capture **model behavior** and **task context**.

---

## 1️⃣ Model Affinity Pathway (Performance Signals)

This pathway measures how confidently AI models process an image.

### 🔹 Inpainting Accuracy (SSIM-based)

Compute inpainting accuracy by comparing the original image with the inpainted (completed) image using **SSIM**:

```bash
python Accuracy.py \
  --original_dir /path/to/img_org \
  --completed_dir /path/to/img_out \
  --output_dir /path/to/output
```
**Observation:**  
AI-generated images tend to be reconstructed with **unusually high accuracy** due to their structural regularity and lack of physical noise.

---

### 🔹 Segmentation Confidence

Compute segmentation confidence by measuring the average per-pixel probability from a semantic segmentation model:

```bash
python confidence.py \
  --input_dir /path/to/images \
  --output_csv results/confidence.csv
```

**Observation:**  
Synthetic images often produce **over-confident and spatially uniform segmentation predictions**, reflecting unnaturally clean boundaries and simplified textures.

---

## 2️⃣ Task Difficulty Pathway (Context Signals)

This pathway estimates how challenging the image *should be* for vision models.

### 🔹 Pixel Naturalness

Compute pixel naturalness by measuring local and global pixel predictability using a learned reconstruction model:

```bash
python Naturalness.py \
  --input_dir /path/to/images \
  --load /path/to/srec_checkpoint.pth \
  --output_csv /path/to/naturalness.csv \
  --gpu_id 0

```

### 🔹 Image Complexity

Compute image complexity by aggregating multiple structural and statistical cues, including edge density, texture variance, entropy, and frequency-domain characteristics:

```bash
python complexity.py \
  --input_dir /path/to/images \
  --output_csv results/complexity.csv
``` 


### 🔹 Feature Fusion & Classification (SVM)

Perform final classification by fusing all extracted features and training an **SVM with an RBF kernel**:

```bash
python Classifier.py \
  --input_xlsx path/to/Extracted_features_dataset.xlsx \
  --output_xlsx path/to/Output.xlsx \
  --test_size 0.5 \
  --seed 42
```
