# 🔥 CNN-Based Forest Fire, Smoke & Non-Fire Image Classifier

**Author:** Aristide Kanamugire  
**Project Type:** Personal Deep Learning Project  
**Focus:** Computer Vision · CNN · Image Classification · AI for Environmental Safety  

🔗 **GitHub Repository:**  
https://github.com/vs98108/mini-project-5  

---

## 📌 1. Project Overview & Motivation

Forest fires pose serious risks to ecosystems, infrastructure, and human life. Early detection of **fire and smoke** from visual data can significantly improve response time and reduce damage.

This project builds a **Convolutional Neural Network (CNN) from scratch** to classify environmental images into three categories:

- 🔥 **Fire**  
- 🌫 **Smoke**  
- 🌲 **Non-Fire**

The goal is to design, train, and evaluate a **robust deep learning model** capable of distinguishing subtle visual patterns in real-world forest environments.

This project demonstrates:

- Practical application of deep learning  
- CNN architecture design  
- Model optimization & generalization  
- Performance evaluation & error analysis  

---

## 🎯 2. Project Objectives

- Build CNN architectures from scratch (no pretrained models)  
- Improve generalization using augmentation and regularization  
- Compare baseline and improved architectures  
- Evaluate model robustness using multiple performance metrics  
- Analyze classification errors  
- Explore architectural improvements using Global Average Pooling  

---

## 🚫 3. Why No Transfer Learning?

This project intentionally avoids transfer learning to:

- Gain deeper understanding of **CNN design principles**  
- Explore **feature extraction learning from raw data**  
- Analyze **generalization behavior without pretrained biases**  

Building models from scratch allows full architectural control and deeper learning insight.

---

## 📂 4. Dataset Description

**Source:** Kaggle – Forest Fire, Smoke & Non-Fire Image Dataset  
https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset  

### Classes

- Fire  
- Smoke  
- Non-Fire  

### Dataset Characteristics

- Varying image resolutions  
- Slight class imbalance  
- Smoke and Non-Fire share strong visual similarity  
- Fire often appears small or distant  
- Lighting and environmental conditions vary widely  

### Preprocessing Pipeline

- Resize images → **128 × 128**  
- Normalize pixel values → **[0,1]**  
- Stratified split:
  - **70% Training**
  - **15% Validation**
  - **15% Testing**
- Fixed random seed → **42** for reproducibility  

---

## 🧠 5. Model Architectures

### 5.1 Baseline CNN

**Architecture:**

- 3 × (Conv2D + ReLU + MaxPooling)  
- Flatten  
- Dense (128 units)  
- Softmax output (3 classes)

**Goal:** Establish a reference baseline.

---

### 5.2 Improved CNN

**Enhancements:**

- Data augmentation:
  - Rotation  
  - Horizontal flip  
  - Zoom  
  - Width/height shift  
- Batch Normalization  
- Dropout  
- EarlyStopping  
- ReduceLROnPlateau  

**Goal:** Reduce overfitting and improve generalization.

---

### 5.3 Architecture Variation (Bonus)

Flatten layer replaced with:

**GlobalAveragePooling2D**

**Advantages:**

- Fewer parameters  
- Reduced overfitting  
- Improved spatial feature representation  
- Better training stability  

---

## ⚙️ 6. Training Configuration

- Optimizer: **Adam**  
- Loss Function: **Categorical Crossentropy**  
- Metrics:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
- Callbacks:
  - EarlyStopping  
  - ReduceLROnPlateau  

---

## 📊 7. Results Summary

### Model Performance

| Model | Accuracy | Precision | Recall | F1-score |
|----------|------------|-------------|----------|-------------|
| Baseline CNN | 97.03% | 97.04% | 97.03% | 97.03% |
| Improved CNN | 95.22% | 95.29% | 95.22% | 95.21% |

---

### Final Model Selection

Although the **Baseline CNN achieved higher raw accuracy**, it showed signs of **overfitting**, including:

- Larger training–validation gap  
- Less stable validation curves  

The **Improved CNN**, despite slightly lower accuracy, demonstrated:

- Better generalization  
- More stable training behavior  
- Reduced confusion between Smoke and Non-Fire  
- Stronger robustness to image variations  

➡️ **The Improved CNN is selected as the final production-ready model.**

---

## 📉 8. Confusion Matrix & Error Analysis

### Observed Patterns

- Thin smoke confused with clouds or fog  
- Small distant fires misclassified as Non-Fire  
- Bright reflections occasionally mistaken for fire  

These findings highlight real-world challenges in wildfire image detection.

---

## 🖼 9. Sample Predictions

Stored in `/images`:

- Correct Fire classification  
- Correct Smoke classification  
- Correct Non-Fire classification  
- Example misclassification  

---

## 🧪 10. Setup & Execution

### Clone Repository

```bash
git clone https://github.com/vs98108/mini-project-5.git
cd mini-project-5
