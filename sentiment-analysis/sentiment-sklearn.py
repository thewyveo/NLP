import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
import numpy as np


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

dataset_combined = pd.read_csv('undersampled_dataset.csv')
dataset_combined.dropna(subset=['text', 'sentiment'], inplace=True)

def run_cross_validation(vectorizer_type, classifier_type, cv_splits=5):
    """
    Run cross-validation with specified vectorizer and classifier
    """
    # Create pipeline
    if vectorizer_type == 'count':
        vectorizer = CountVectorizer(min_df=2)
    else:
        vectorizer = TfidfVectorizer(min_df=2)
    
    if classifier_type == 'nb':
        classifier = MultinomialNB()
    else:
        classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    # Create stratified k-fold cross-validator
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    
    # Run cross-validation
    X = dataset_combined['text']
    y = dataset_combined['sentiment']
    
    print(f"\nRunning {cv_splits}-fold CV with {vectorizer_type} vectorizer and {classifier_type} classifier")
    
    # Get cross-validation scores
    cv_scores = cross_val_score(
        pipeline, 
        X, 
        y, 
        cv=skf,
        scoring='accuracy',
        n_jobs=-1  # Use all available cores
    )
    
    print(f"\nCross-validation accuracy scores: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    # Additional metrics from one full training run for detailed reporting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print("\nDetailed metrics on holdout set:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return pipeline

def classify_test_set(model, test_file_path='sentiment-topic-test.tsv', print_samples=True):
    """
    Classify the test set and show detailed predictions
    """
    # Load test data
    test_data = pd.read_csv(test_file_path, sep='\t')
    
    # Get predictions
    predictions = model.predict(test_data['sentence'])
    print(predictions)
    test_data['predicted'] = predictions
    
    # Print each sentence with true/predicted labels
    if print_samples:
        print("\nDetailed Predictions:")
        print("-" * 80)
        for i, row in test_data.iterrows():
            print(f"\sentence: {row['sentence']}")
            print(f"True: {row['sentiment']} | Predicted: {row['predicted']}")
            print("-" * 80)
    
    # Evaluation metrics
    print("\nOverall Evaluation:")
    print(classification_report(test_data['sentiment'], test_data['predicted']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_data['sentiment'], test_data['predicted']))
    
    return test_data

# Example usage in main:
if __name__ == "__main__":
    # Train model
    print("Training model...")
    model = run_cross_validation('tfidf', 'lr')
    
    # Evaluate on test set with detailed output
    print("\nEvaluating on test set...")
    results = classify_test_set(model)