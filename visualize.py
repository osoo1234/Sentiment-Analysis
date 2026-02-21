import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_visuals(file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(file_path)
    
    # 1. Sentiment Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='final_sentiment', data=df, palette='viridis')
    plt.title('Overall Sentiment Distribution')
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
    plt.close()
    
    # 2. Sentiment by Airline
    if 'airline' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='airline', hue='final_sentiment', data=df)
        plt.title('Sentiment by Airline')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(output_dir, 'sentiment_by_airline.png'))
        plt.close()
        
    # 3. Topic Impact
    if 'topic_id' in df.columns:
        plt.figure(figsize=(10, 6))
        topic_counts = df['topic_label'].value_counts()
        topic_counts.plot(kind='barh')
        plt.title('Top Topics by Volume')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'topic_distribution.png'))
        plt.close()

if __name__ == "__main__":
    final_data = r"d:\Desktop\Rivoo\Sentiment Analysis\data\tweets_final.csv"
    output_visuals = r"d:\Desktop\Rivoo\Sentiment Analysis\reports\visuals"
    if os.path.exists(final_data):
        generate_visuals(final_data, output_visuals)
        print(f"Visuals saved to {output_visuals}")
