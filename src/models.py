from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

class SentimentModel:
    def __init__(self, model_type='logistic'):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'svm':
            self.model = SVC(probability=True)
        elif model_type == 'mlp':
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
        self.model_type = model_type

    def train(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        return metrics, y_test, y_pred

    def predict(self, texts):
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict(X_vec)

    def save(self, model_path, vec_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vec_path)

def plot_confusion_matrix(cm, labels, output_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()
