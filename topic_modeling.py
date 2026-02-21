import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import os

def extract_topics(file_path, output_path, n_topics=5):
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Fill NaN just in case
    texts = df['cleaned_text'].fillna("")
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print(f"Fitting LDA with {n_topics} topics...")
    from sklearn.decomposition import LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)
    
    # Assign topics to records
    topic_results = lda.transform(tfidf_matrix)
    df['topic_id'] = topic_results.argmax(axis=1)
    
    # Get top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for i, topic in enumerate(lda.components_):
        topic_keywords[i] = [feature_names[index] for index in topic.argsort()[-10:]]
    
    print("Topic Keywords identified:")
    for i, keywords in topic_keywords.items():
        print(f"Topic {i}: {', '.join(keywords)}")

    # Map topic IDs to descriptive labels based on keywords (Manual/Heuristic)
    # For now, we'll keep it as "Topic X" or try to name the top keyword.
    df['topic_label'] = df['topic_id'].apply(lambda x: f"Topic {x}: " + ", ".join(topic_keywords[x][:3]))
    
    # Calculate Reputation Impact
    # Impact = Volume of Negative sentiment in that topic
    topic_sentiment = df.groupby('topic_id')['final_sentiment'].value_counts(normalize=True).unstack().fillna(0)
    
    def get_impact_level(topic_id):
        neg_ratio = topic_sentiment.loc[topic_id].get('negative', 0)
        if neg_ratio > 0.7: return "High"
        if neg_ratio > 0.4: return "Medium"
        return "Low"
    
    df['reputation_impact'] = df['topic_id'].apply(get_impact_level)
    
    print(f"Saving results to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    base_path = r"d:\Desktop\Rivoo\Sentiment Analysis\data"
    
    # Process Tweets
    tweets_sentiment = os.path.join(base_path, "tweets_sentiment.csv")
    if os.path.exists(tweets_sentiment):
        extract_topics(tweets_sentiment, os.path.join(base_path, "tweets_final.csv"))
