# 🌾 Precision Farming and Disease Detection using Machine Learning

A smart agriculture system that empowers farmers with intelligent crop recommendations and plant disease detection using advanced machine learning algorithms. This project aims to boost crop productivity, reduce losses, and promote sustainable farming practices.

---

## 🚀 Overview

This project includes two core modules:

1. **🌱 Crop Recommendation System**  
   Uses soil parameters (NPK, pH, humidity, rainfall, etc.) to recommend the most suitable crop using a Random Forest Classifier.

2. **🦠 Plant Disease Detection System**  
   Identifies diseases in plant leaves using a Convolutional Neural Network (CNN) model trained on image datasets and suggests appropriate cures.

---

## 🧠 Problem Statement

Develop a machine learning-based precision farming system that recommends optimal crops, detects and prevents plant diseases, and provides actionable insights to improve yield. The goal is to assist farmers in making informed decisions by analyzing soil and crop data.

---

## 📊 Tech Stack

- **Languages:** Python  
- **ML Models:** Random Forest, CNN (Convolutional Neural Network)  
- **Libraries:** NumPy, Pandas, Scikit-learn, TensorFlow/Keras, OpenCV, Matplotlib  
- **Tools:** Jupyter Notebook, Google Colab, VS Code

---

## 📁 Dataset

- **Crop Recommendation:** CSV file containing soil content, temperature, pH, humidity, rainfall, and crop labels.
- **Disease Detection:** Image dataset of healthy and diseased plant leaves (e.g., from PlantVillage dataset).

---

## 🧪 How It Works

### ✅ Crop Recommendation Module
- Input: Soil parameters
- Model: Random Forest Classifier
- Output: Suggested crop suitable for the given soil conditions

### 🖼️ Disease Detection Module
- Input: Image of a plant leaf
- Model: Convolutional Neural Network (CNN)
- Output: Disease classification and suggested cure

---

## 🧾 Installation

```bash
git clone https://github.com/your-username/precision-farming-ml.git
cd precision-farming-ml
pip install -r requirements.txt
