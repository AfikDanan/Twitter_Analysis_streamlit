import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import streamlit as st
from helpers import *

st.title("Anti-Semitism vs. Anti-Zionism: A Twitter Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Know the data", "Models & Clustering", "Classify Tweets", "Insights & Conclusion"])

with tab1:
    df_jews = pd.read_csv('data/clean_jews_tweets.csv')
    df_zionist = pd.read_csv('data/clean_zionism_tweets.csv')

    df_jews_tagged = pd.read_csv("data/clean_tagged_jews_tweets.csv")
    df_zionist_tagged = pd.read_csv("data/clean_tagged_zionism_tweets.csv")

    data = pd.concat([df_jews, df_zionist])
    data_tagged = pd.concat([df_jews_tagged, df_zionist_tagged])  

    data_filtered = data_tagged[data_tagged['label'] != 'neutral']
    vectorizer = TfidfVectorizer().fit_transform(data_filtered['processed_text'])
    
# Sidebar options
    st.sidebar.title('Visualization Options')
    option = st.sidebar.selectbox('Select Visualization', 
                                  ['All',
                                    'Class Counts', 
                                    'Emotion Distribution',
                                    'Word Cloud', 
                                    'PCA Visualization', 
                                    'Cosine Similarity Heatmap', 
                                ])

    # Display selected visualization
    if option == 'All':
        visualize_class_counts(data_filtered)
        emotion_dist_plot(data_tagged)
        plot_wordcloud(data_filtered)
        tweet_pca_visualization(data_filtered, vectorizer)
        display_cosine_similarity_heatmap(data_filtered, vectorizer)
        
    elif option == 'Cosine Similarity Heatmap':
        display_cosine_similarity_heatmap(data_filtered, vectorizer)
    elif option == 'PCA Visualization':
        tweet_pca_visualization(data_filtered, vectorizer)
    elif option == 'Word Cloud':
        plot_wordcloud(data_filtered)
    elif option == 'Class Counts':
        visualize_class_counts(data_filtered)
    elif option == 'Emotion Distribution':
        emotion_dist_plot(data_tagged)
    
    
with tab2:
    st.header("Modeling and Clustering")
    y = data_filtered['label']
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    X_vec = vectorizer.fit_transform(data_filtered['processed_text'])
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_vec, data_filtered['label'], test_size=0.2, random_state=42)


    # Implementing SVM
    st.markdown("## SVM Classification Report")
    y_pred_svm, svm_model =svm_model_func(X_train, X_test, y_train, y_test)
    confusion_matrix_plot(y_test, y_pred_svm, svm_model)
    st.markdown("""
    ### Key Observations:
    The SVM model shows good overall performance with an accuracy of **80%**. However, it has a tendency to misclassify **anti-Semitic** tweets as **anti-Zionist** more frequently. This suggests the need for further optimization in distinguishing between these two categories to reduce misclassification rates.
    """)

    # Implementing Decision Tree
    st.header("Decision tree Classification Report")    
    y_pred_dt, dt_model= decision_tree_model_func(X_train, X_test, y_train, y_test)
    confusion_matrix_plot(y_test, y_pred_dt, dt_model)
    st.markdown("""
    - **Overall Accuracy**: **82%**
    - Better than Naive Bayes, similar to SVM

    **Conclusion**: The Decision Tree model effectively differentiates between the categories, though some overlap remains.
    """)

    # Implementing Naive Bayes
    st.header("Naive Bayes Classification Report")
    y_pred_nb, nb_model = naive_bayes_model_func(X_train, X_test, y_train, y_test)
    confusion_matrix_plot(y_test, y_pred_nb, nb_model)
    st.write("""
    Key takeaways from Naive Bayes Classification:

    1. Low overall accuracy (57%), only slightly better than random guessing.
    2. Strong bias towards classifying tweets as anti-Zionist.
    3. High misclassification of anti-Semitic tweets (16 out of 19) as anti-Zionist.
    4. Poor performance compared to the SVM model and the Decision tree.

    This model is unreliable for distinguishing between anti-Semitic and anti-Zionist content.
    """)
      
    
    # Implementing logistic regression
    st.header("Logistic Regression Classification Report")
    y_pred_lr, lr_model = LogisticRegression_model_func(X_train, X_test, y_train, y_test)
    confusion_matrix_plot(y_test, y_pred_lr, lr_model)
    st.markdown("""
    - **Overall Performance**:
    - Accuracy: **60%**

    **Conclusion**: The model perfectly recalls anti-Zionist content but struggles with anti-Semitic content, leading to overclassification as anti-Zionist and a trade-off between high precision and low recall.
    """)
    



with tab3: 
    st.header("Classify Tweets")
    tweet = st.text_input("Enter a tweet to classify:")
    if tweet:
        prediction = classify_tweet(tweet, svm_model, vectorizer)
        st.markdown(f"### Prediction: {prediction} tweet")

with tab4:
    st.header("Insights & Conclusion")
    st.markdown("""
    #### Key Insights:

    - Class Distribution: The dataset contains more anti-Zionist tweets than anti-Semitic tweets, with a notable overlap between the two classes.
    - Model Performance: SVM and Decision Tree models perform better in distinguishing between anti-Zionist and anti-Semitic tweets compared to Naive Bayes and Logistic Regression.
    - Cosine Similarity: A high average similarity score between anti-Zionist and anti-Semitic tweets indicates a significant overlap in language and themes.
    - Challenges: Misclassification of anti-Semitic tweets as anti-Zionist is a common issue across models, suggesting the need for more nuanced features or additional data for better classification.
    
    #### Conclusion:

    - Model Evaluation: SVM and Decision Tree models are more effective in classifying tweets compared to Naive Bayes and Logistic Regression. However, all models face difficulty due to the semantic overlap between anti-Zionist and anti-Semitic language.
    - Overlap in Content: The high cosine similarity score and misclassification rates reveal that anti-Zionist and anti-Semitic tweets frequently share common language, expressions, and themes. This overlap complicates the classification task and reflects the ambiguity and convergence in online discourse.
    - Struggles in Distinguishing Between Anti-Semitism and Anti-Zionism on Twitter:
        - Ambiguity in Language: On Twitter, anti-Zionist and anti-Semitic content often uses overlapping terminology (e.g., references to Israel, Zionism, and Jewish identity), making it hard to distinguish intent and context. Some tweets labeled as anti-Zionist can subtly incorporate anti-Semitic themes, while others remain strictly political. The models struggle with these nuances.
        - Contextual Dependence: Many tweets lack sufficient context for clear categorization. Short text formats, sarcasm, and implicit biases further challenge the models' ability to accurately classify content.
        - Initial Sentiment Analysis: Initially, sentiment analysis was applied to classify the tweets; however, it proved insufficient for distinguishing between anti-Zionist and anti-Semitic content due to the complex and overlapping nature of the language used.
        - News Reports: Some tweets were news reports, which are neither anti-Semitic nor anti-Zionist. This inclusion further complicates the classification task, as they do not fit neatly into either category.
        
    This analysis underlines the complex challenge of differentiating anti-Semitic and anti-Zionist content on social media, highlighting the necessity for sophisticated models, richer data, and innovative methods to address these challenges effectively.
    """)