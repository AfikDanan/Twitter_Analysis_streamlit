# Anti-Semitism vs. Anti-Zionism: A Twitter Analysis

## Overview

This Streamlit application analyzes tweets to distinguish between anti-Semitic and anti-Zionist content. It uses machine learning models, clustering techniques, and various visualizations to explore the overlap and distinctions between these categories. The project aims to uncover key insights and challenges in classifying such content.

## Features

### 1. Know the Data
- **Data Overview**: Displays datasets of tweets labeled as anti-Semitic or anti-Zionist.
- **Visualizations**:
  - Class Counts
  - Emotion Distribution
  - Word Cloud
  - PCA Visualization
  - Cosine Similarity Heatmap

### 2. Models & Clustering
- **Machine Learning Models**:
  - **SVM (Support Vector Machine)**: Demonstrates strong overall performance but shows a tendency to misclassify anti-Semitic tweets as anti-Zionist.
  - **Decision Tree**: Provides comparable performance to SVM with good differentiation between categories.
  - **Naive Bayes**: Low accuracy and bias towards classifying tweets as anti-Zionist, resulting in high misclassification rates.
  - **Logistic Regression**: Struggles with distinguishing between categories due to overlapping language.

### 3. Classify Tweets
- **Real-time Classification**: Allows users to input a tweet and classify it using the trained SVM model.

### 4. Insights & Conclusion
- **Key Insights**:
  - Dataset contains more anti-Zionist tweets, with notable overlap between the two classes.
  - SVM and Decision Tree models are more effective compared to Naive Bayes and Logistic Regression.
  - High cosine similarity between anti-Zionist and anti-Semitic tweets suggests significant overlap in language and themes.
  - Common misclassification of anti-Semitic tweets as anti-Zionist indicates a need for nuanced features or additional data.
- **Conclusions**:
  - Models face challenges due to the semantic overlap between anti-Zionist and anti-Semitic language.
  - Ambiguity and contextual dependence in tweets complicate classification.
  - Inclusion of news reports adds complexity, as they do not fit neatly into either category.

## Getting Started

### Prerequisites
- Python 3.8 or above
- Required libraries: `pandas`, `scikit-learn`, `streamlit`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/antisemitism-vs-antizionism-analysis.git
