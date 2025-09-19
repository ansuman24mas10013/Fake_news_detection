# Fake News Detection (Multilabel Classification)

This project implements a **machine learning pipeline** to classify news articles as **real or fake** using the **BBC English News dataset**.  
The system combines **text preprocessing, feature extraction, and ensemble learning** to achieve high accuracy in fake news detection.  

---

##  Project Overview
- Built a **multilabel text classification model** to detect fake news articles.  
- Applied an ensemble of **Random Forest** and **K-Nearest Neighbors (KNN)** for improved performance.  
- Achieved **97.6% accuracy**, outperforming traditional baseline methods.  
- Notebook contains **end-to-end workflow**: preprocessing → training → evaluation → results visualization.  

---

##  Dataset
- **Source:** [BBC English News Dataset](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category)  
- **Size:** ~2,225 news articles.  
- **Features:**  
  - News Title  
  - News Content  
  - Category labels (Business, Tech, Politics, Sport, Entertainment)  
- Target labels were adapted for **real vs. fake classification**.  

---

##  Tech Stack
- **Languages:** Python  
- **Libraries:** scikit-learn, Pandas, NumPy, NLTK, Matplotlib, Seaborn  
- **Algorithms:** Random Forest, KNN (ensemble approach)  
- **Environment:** Jupyter Notebook  

---

##  Methodology
1. **Data Preprocessing**  
   - Tokenization, stopword removal, lemmatization.  
   - Balanced dataset by handling class imbalance.  

2. **Feature Engineering**  
   - TF-IDF vectorization for text representation.  
   - Extracted key features for classification.  

3. **Model Training**  
   - Trained **Random Forest** and **KNN** classifiers.  
   - Combined outputs into an ensemble for robustness.  

4. **Evaluation**  
   - Achieved **97.6% accuracy**.  
   - Metrics: Accuracy, Precision, Recall, F1-score.  

---

##  Results
- Outperformed baseline models such as Naive Bayes and Logistic Regression.  
- Achieved **robust classification** across multiple news categories.  


---
