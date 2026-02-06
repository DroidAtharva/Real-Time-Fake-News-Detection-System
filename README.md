# Fake News Detection System 📰🤖

A machine learning–based system designed to classify news articles as **Real** or **Fake** using Natural Language Processing (NLP) techniques and supervised learning algorithms.

This project addresses the growing challenge of misinformation on digital platforms by analyzing textual content and identifying deceptive patterns in news articles.

---

## 📌 Problem Statement

In the era of digital communication, the rapid spread of fake news poses serious threats to public trust, political stability, and informed decision-making. Fake news shared through social media and online news platforms can mislead users and influence opinions unfairly.

The objective of this project is to develop a **reliable and scalable Fake News Detection System** that automatically classifies news articles as *Real* or *Fake* based on their textual content.

---

## 🧠 Project Overview

The system uses **Natural Language Processing (NLP)** for text preprocessing and **Machine Learning** for classification.  
Textual features are extracted using **TF-IDF Vectorization**, and a **Random Forest Classifier** is trained on labeled news data to identify misleading content.

The system supports both:
- Manual news headline input
- API-based live news input (conceptual / extendable)

---

## ⚙️ System Workflow

1. **Data Collection**
   - Real and Fake news datasets (`Fake.csv`, `True.csv`)

2. **Data Cleaning**
   - Removal of punctuation, stopwords, and irrelevant symbols
   - Tokenization and text normalization

3. **Feature Extraction**
   - TF-IDF (Term Frequency–Inverse Document Frequency)

4. **Model Training**
   - Random Forest Classifier (Supervised Learning)

5. **Model Evaluation**
   - Accuracy and performance metrics

6. **Prediction**
   - Output displayed as **FAKE** or **REAL**

---

## 🧩 Block Diagram (Logical Flow)

