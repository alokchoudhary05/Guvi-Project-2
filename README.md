# Movie Review Sentiment Analyzer
A machine learning project that classifies movie reviews as **Positive** or **Negative** using **Natural Language Processing (NLP)** and **Logistic Regression**.  
The project is implemented in Python, with a full pipeline from data preprocessing to model training and deployment using **Streamlit**.

---

## Problem Statement
Manually reading and classifying hundreds of reviews is time-consuming, inconsistent, and prone to bias.  
This project aims to automate the process, allowing for **fast and accurate sentiment classification**.

---

## Objective
To build an **automated sentiment analysis model** that can:
- Process large datasets of movie reviews.
- Predict sentiment (Positive/Negative) based on review text.
- Be deployed as an interactive **Streamlit web app**.

---

## Project Workflow
1. **Load Dataset** – Import movie review dataset with sentiment labels.  
2. **Preprocess Text** – Lowercasing, punctuation removal, stopword removal.  
3. **Feature Extraction** – Convert text into numerical format using **TF-IDF Vectorizer**.  
4. **Model Training** – Train Logistic Regression model on training data.  
5. **Evaluation** – Test model performance with Accuracy, F1 Score, Confusion Matrix.  
6. **Deployment** – Deploy model with a user-friendly **Streamlit interface**.

---

## Tech Stack
- **Python**
- **Pandas**, **NumPy** – Data handling
- **NLTK** – Text preprocessing
- **Scikit-learn** – TF-IDF, Logistic Regression, Model Evaluation
- **Matplotlib**, **Seaborn** – Data visualization
- **Streamlit** – Web app deployment

---

## Dataset
The dataset used is a **synthetic dataset** of 1500+ movie reviews labeled as Positive or Negative for demonstration purposes.

> Note: Due to the dummy dataset, models achieved unusually high accuracy (close to 100%), which will be lower with real-world data.

---

## Model & Techniques Used
- **Text Preprocessing**
  - Lowercasing
  - Stopword removal
  - Punctuation removal
- **Feature Extraction**
  - TF-IDF Vectorizer
- **Machine Learning Model**
  - Logistic Regression (Primary)
  - Naive Bayes, SVM, Random Forest, Decision Tree, KNN (Tested for comparison)

---

## Results
- **Primary Model**: Logistic Regression
- **Test Accuracy**: ~100% (due to dummy dataset)
- **Key Learning**:
  - End-to-end ML project implementation
  - Impact of text preprocessing on model performance
  - Use of TF-IDF in NLP
  - Model deployment with Streamlit

---

## Thanks

---
