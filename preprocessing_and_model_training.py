import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

class NewsClassifier:
    def __init__(self, sample_size=100000):
        """
        Initialize the News Classifier
        
        Parameters:
        -----------
        sample_size : int, optional
            Number of samples to use (e.g 10,000)
        """
        self.sample_size = sample_size
        self.vectorizer = None
        self.classifier = None
        self.label_encoder = None
    
    def clean_text(self, text):
        """
        Clean input text
        
        Parameters:
        -----------
        text : str
            Input text to clean
        
        Returns:
        --------
        str: Cleaned text
        """
        # Convert to lowercase using Python string method
        text = text.lower() if isinstance(text, str) else str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', '', text)
        
        return text
    
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the news dataset with sampling
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing news data
        
        Returns:
        --------
        pandas.DataFrame: Processed dataframe
        """
        # Load the dataset
        df = pd.read_csv(file_path)
        
        # Remove duplicates first
        df.drop_duplicates(subset=['headline', 'short_description'], inplace=True)
        
        # Sample data
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
        
        # Combine headline and short description
        df['combined_text'] = df['headline'].astype(str) + ' ' + df['short_description'].astype(str)
        
        # Clean text
        df['cleaned_text'] = df['combined_text'].apply(self.clean_text)
        
        return df
    
    def extract_features(self, df):
        """
        Extract features using TF-IDF Vectorization
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed dataframe
        
        Returns:
        --------
        tuple: Vectorized features and encoded labels
        """
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['category'])
        
        return X, y
    
    def train_model(self, X_train, y_train):
        """
        Train a logistic regression model
        Parameters:
        -----------
        X_train : scipy.sparse matrix
            Training feature matrix
        y_train : numpy.ndarray
            Training labels
        
        Returns:
        --------
        Trained classification model
        """
        # Use Logistic Regression
        self.classifier = LogisticRegression(
            max_iter=500,
            multi_class='multinomial',
            solver='lbfgs',
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        return self.classifier
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the classification model
        Parameters:
        -----------
        X_test : scipy.sparse matrix
            Test feature matrix
        y_test : numpy.ndarray
            Test labels
        
        Returns:
        --------
        dict: Classification metrics
        """
        # Predictions
        y_pred = self.classifier.predict(X_test)
        
        # Classification Report
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion Matrix Visualization
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True,     
            fmt='d', 
            xticklabels=self.label_encoder.classes_, 
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
        
        return report
    
    def predict_category(self, text):
        """
        Predict category for a new piece of text
        Parameters:
        -----------
        text : str
            Input text to classify
        
        Returns:
        --------
        str: Predicted category
        """
        # Ensure model is trained
        if self.classifier is None or self.vectorizer is None or self.label_encoder is None:
            raise ValueError("Model must be trained before prediction")
        
        # Clean and vectorize the text
        cleaned_text = self.clean_text(text)
        text_vectorized = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.classifier.predict(text_vectorized)
        
        # Decode the prediction
        return self.label_encoder.inverse_transform(prediction)[0]
    
    def train(self, file_path):
        """
        Complete training workflow
        Parameters:
        -----------
        file_path : str
            Path to the dataset
        
        Returns:
        --------
        dict: Training results
        """
        # Load and preprocess data
        df = self.load_and_preprocess_data(file_path)
        
        # Extract features
        X, y = self.extract_features(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        self.train_model(X_train, y_train)
        
        # Evaluate the model
        evaluation_report = self.evaluate_model(X_test, y_test)
        
        return {
            'data_size': len(df),
            'features': X.shape[1],
            'evaluation_report': evaluation_report
        }

# usage
def main():
    # Create classifier instance
    classifier = NewsClassifier(sample_size=100000)
    
    # Train the model
    results = classifier.train('nlp\\news.csv')
    print("Training Results:", results)
    
    # prediction
    try: 
        sample_text = "We're Pandemic Experts. Here's What Worries Us Most About Bird Flu."
        predicted_category = classifier.predict_category(sample_text)
        print(f"Predicted Category: {predicted_category}")

    except Exception as e:
        print(f"Prediction error: {e}")

if __name__ == "__main__":
    main()