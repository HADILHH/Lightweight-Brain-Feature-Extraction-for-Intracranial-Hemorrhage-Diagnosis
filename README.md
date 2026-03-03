# Lightweight Brain Feature Extraction for Intracranial Hemorrhage Diagnosis  
## Using Multi-Algorithm Hybrid Features  


## Abstract

Intracranial Hemorrhage (ICH) is a life-threatening emergency that can cause severe neurological damage or death if not detected early. Deep learning methods have shown excellent performance in automatic ICH detection from CT images, but their high computational cost and lack of interpretability limit their clinical usability.

This project proposes a lightweight hybrid feature extraction framework that combines classical image processing and machine learning methods for fast and interpretable ICH diagnosis. The proposed system integrates efficient algorithms to extract intensity, texture, edge, and frequency-based features from brain CT scans.

Three public datasets — RSNA-ICH, CQ500, and PhysioNet Head CT — are used for training and validation. The objective is to achieve high diagnostic accuracy, strong cross-dataset generalization, and low computational cost suitable for real-time clinical screening.

---

## 1. Introduction

Intracranial Hemorrhage (ICH) occurs due to bleeding inside the skull, often caused by trauma, hypertension, or aneurysm rupture. Early detection and accurate classification of hemorrhage types — epidural (EDH), subdural (SDH), subarachnoid (SAH), intraparenchymal (IPH), and intraventricular (IVH) — are critical for timely medical intervention.

Although deep neural networks have achieved high performance, they demand significant computational resources and lack interpretability. This project aims to develop a hybrid lightweight approach that extracts interpretable and computationally efficient features for automatic ICH diagnosis from CT scans.

---

## 2. Objectives

- Develop efficient algorithms to extract intensity, texture, edge, and frequency-based features from brain CT images.
- Design a hybrid feature vector combining multiple types of descriptors.
- Train and compare multiple classifiers (SVM, ANN, AdaBoost, Random Forest) for ICH detection and classification.
- Validate the proposed framework on multiple public datasets and evaluate robustness and computational efficiency.

---

## 3. Methodology Overview

The proposed pipeline consists of:

1. CT image preprocessing  
2. Hybrid feature extraction  
3. Feature normalization and selection  
4. Multi-algorithm classification  
5. Performance evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC)

---

## 4. Project Structure
ICH-Project/
│
├── preprocessing/
├── extract_features.py
├── train_model.py
├── predict_single_image.py
├── ich_gui.py
├── requirements.txt
└── README.md

---

## 5. Models Implemented

- Support Vector Machine (SVM)
- Artificial Neural Network (ANN)
- Random Forest (RF)
- Logistic Regression
- CNN (for comparison)

---

## 6. Installation

```bash
pip install -r requirements.txt
python train_model.py
python predict_single_image.py


