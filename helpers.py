import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from wordcloud import WordCloud

import streamlit as st


def confusion_matrix_plot(y_test, y_pred, model):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for ' + str(model))
    st.pyplot(plt)   
   
def svm_model_func(X_train, X_test, y_train, y_test):
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred_svm = svm_model.predict(X_test)

    # Classification Report
    report_svm = classification_report(y_test, y_pred_svm)
    st.text("SVM Classification Report:")
    st.text(report_svm)
    return y_pred_svm, svm_model

def naive_bayes_model_func(X_train, X_test, y_train, y_test):
    # Training Naive Bayes model
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred = nb_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    st.text("Classification Report:")
    st.text(report) 
    return y_pred, nb_model
    
    
def decision_tree_model_func(X_train, X_test, y_train, y_test): 
    # Implementing Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred_dt = dt_model.predict(X_test)

    # Classification Report
    report_dt = classification_report(y_test, y_pred_dt)
    st.text("Decision Tree Classification Report:")
    st.text(report_dt)
    return y_pred_dt, dt_model
    
    
    
def LogisticRegression_model_func(X_train, X_test, y_train, y_test):
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    st.text("Logistic Regression Classification Report:")
    st.text(report)
    
    return y_pred, model

def display_cosine_similarity_heatmap(data_filtered, vectorizer):
    st.markdown("## Cosine Similarity Heatmap Analysis")
    class_labels = data_filtered['label'].unique()
    class_vectors = {}
    for label in class_labels:
        class_vectors[label] = vectorizer[data_filtered['label'] == label].mean(axis=0).A1
    class_vectors_stacked = np.stack(list(class_vectors.values()))
    cosine_sim_matrix = cosine_similarity(class_vectors_stacked)
    st.write(f'Average Cosine Similarity between Anti-Zionism and Anti-Semitism tweets: {cosine_sim_matrix.mean()}')
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=class_labels, columns=class_labels)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cosine_sim_df, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Cosine Similarity Between Classes")

    # Display the plot in Streamlit
    st.pyplot(plt)
    
    st.markdown("""
    This heatmap shows the **cosine similarity** between **anti-Zionist** and **anti-Semitic** tweets.

    ### Key Observations:
    - **Average Similarity**: **0.699**, indicating a strong correlation between these two classes of tweets.
    - **Perfect Similarity (1.00)**: Within each class, tweets exhibit perfect similarity, showing homogeneity in content.
    - **Cross-Class Similarity**: **0.40**, suggesting a significant overlap in language and themes between anti-Zionist and anti-Semitic content on social media.

    ### Conclusion:
    The high average similarity score and the notable cross-class similarity indicate that anti-Zionist and anti-Semitic tweets often share common language and themes, reflecting a considerable degree of overlap in social media discourse.
    """)
    
    
def tweet_pca_visualization(data_filtered, vectorizer):
    st.markdown("# PCA Visualization of Tweets")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(vectorizer.toarray())

    embedding  = pd.DataFrame(pca_result, columns=["x", "y"])
    embedding['label'] = data_filtered['label'].values

    # Plotting
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='x', y='y', hue='label', palette='viridis', data=embedding)
    plt.title(f'PCA Visualization of Tweets')
    # Display the plot in Streamlit
    st.pyplot(plt)
    st.markdown("""
    This scatter plot represents a **Principal Component Analysis (PCA)** of tweets, visualizing the two main components that capture the most variance in the dataset. Each point represents a tweet, reduced to two dimensions using PCA.

    ### Observations
    - **Labels**: The points are colored according to their labels:
        - **Anti-Zionist** (blue)
        - **Anti-Semitic** (green)
    - **Clusters**:
        - Tweets labeled as **anti-Zionist** (blue) and **anti-Semitic** (green) show some overlap, indicating similarity in their principal components.
    - The distribution of points suggests there are both distinct clusters and some mixing between the two classes.

    ### Interpretation
    The visualization helps identify patterns or similarities between different tweet labels. Overlapping regions may suggest that the features used for PCA are not fully separating these classes, indicating potential challenges in classification or the need for further feature engineering.
    """)
    
def plot_wordcloud(data_tagged):
    st.markdown("# Word Cloud Analysis")
    wordcloud = WordCloud(width=800, height=400).generate(" ".join(data_tagged['processed_text']))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
    st.markdown("This word cloud visually represents the most frequently occurring words in a dataset related to geopolitical and social discussions. The words reflect a focus on conflict, political issues, and ideologically charged language.")
    st.markdown("### Prominent Words")
    st.markdown("""
    - **People**: A key word, indicating that discussions often revolve around individuals or groups.
    - **Zionist**: Frequently used, suggesting a focus on ideologies and political groups.
    - **Jew, Israel, Palestinian, Gaza**: Highlighted in larger fonts, showing these are common and central terms in the dataset.
    """)

    st.markdown("### Other Significant Terms")
    st.markdown("""
    - **Genocide, war, terrorist, crime, hate, murder**: Words related to conflict and violence are prominent, indicating a dataset focused on contentious and politically sensitive topics.
    - **Support, right, state**: Terms related to political stances and governance issues.
    """)

    st.markdown("The word cloud provides insights into the language and themes prevalent in the data, showing an emphasis on conflict, political discourse, and social issues.")

    
def visualize_class_counts(data_tagged):
    st.header("Class Distribution in the tagged data")    
    class_counts = data_tagged['label'].value_counts()
    st.bar_chart(class_counts)
    st.write("* anti-Zionist: The most frequent class, with around 100 occurrences.")
    st.write("* anti-Semitic: The second most frequent, with about 80 occurrences.")
    st.write("* neutral: The least frequent, with around 50 occurrences.")
    st.write("This distribution suggests that \"anti-Zionist\" is the most commonly tagged class, followed by \"anti-Semitic,\" while \"neutral\" is the least common. The visualization helps understand the balance or imbalance among the tagged categories in the dataset.\n\n")
    
def emotion_dist_plot(df):
    st.markdown("## Emotion Distribution Analysis")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='emotion', data=df, hue='label',palette='viridis')
    plt.title(f'Emotions Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    st.pyplot(plt)
    
    st.write(f"- **Disgust** is the most common emotion in the dataset, suggesting that the majority of the content reflects this emotion.")
    st.write(f"- The category **'others'** has a significant presence, indicating a variety of other emotions, which could include neutral or mixed emotions.")
    st.write(f"- The emotion **anger** also has a noticeable count, indicating that a substantial portion of the dataset expresses anger.")
    st.write(f"- Overall, the distribution shows a strong skew towards negative emotions like disgust and anger, with only a minor presence of fear and sadness. This suggests that the dataset might be composed of content that evokes strong negative reactions, with disgust being the most prevalent.")

    
    # Function to classify new tweets
def classify_tweet(tweet, model, vectorizer):
    tweet_tfidf = vectorizer.transform([tweet])
    prediction = model.predict(tweet_tfidf)
    return prediction[0]