import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import nltk
from tqdm import tqdm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
'''
DATA PREROCESSING (IS NOT COMPLETE, see sentiment-data-preprocessing.ipynb)

# Loading and preparing the Stanford Sentiment Treebank dataset
sentences_stf = pd.read_csv('stanfordSentimentTreebank\datasetSentences.txt', sep='\t')
sentiments_stf = pd.read_csv('stanfordSentimentTreebank\sentiment_labels.txt', sep='|', engine='python')
sentiments_stf.columns = ['sentence_index', 'sentiment_value']
dataset_stf = pd.merge(sentences_stf, sentiments_stf, on='sentence_index')
dataset_stf = dataset_stf.rename(columns={'sentence': 'text', 'sentiment_value': 'rating'})
dataset_stf.drop('sentence_index', axis=1, inplace=True)

# Loading and preparing the Amazon Reviews dataset
dataset_reviews = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023", 
    "raw_review_Books", 
    split="full[:100000]", 
    trust_remote_code=True
    )
dataset_reviews = pd.DataFrame(dataset_reviews)
dataset_reviews = dataset_reviews[['rating', 'text']]
min_count = dataset_reviews['rating'].value_counts().min()
dataset_reviews = dataset_reviews.groupby('rating').apply(lambda x: x.sample(n=min_count, random_state=42), include_groups=False).reset_index(drop=True)

# Combining the datasets
dataset_stf['rating'] = (dataset_stf['rating'] * 5).clip(1, 5).round()
dataset_combined = pd.concat([dataset_reviews, dataset_stf], ignore_index=True)
dataset_combined = dataset_combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

def rating_to_sentiment(rating):
    if rating == 4.0 or rating == 5.0 or rating == 3.0:
        return 'positive'
    elif rating == 1.0:
        return 'negative'
    else:
        return 'neutral'
'''

# Loading and preparing the VU Sentiment dataset
VU_test_data = pd.read_csv('sentiment-topic-test.tsv', sep='\t')
VU_test_data.drop('sentence_id', axis=1, inplace=True)

# Loading the combined dataset amazon reviews and Stanford Sentiment Treebank
dataset_combined = pd.read_csv('dataset_combined.csv')
dataset_combined.dropna(subset=['text', 'sentiment'], inplace=True)

# Initialize models
nlp = spacy.load("en_core_web_sm")
vader_model = SentimentIntensityAnalyzer()
nltk.download('vader_lexicon')

def run_vader(sentence, lemmatize=True):
    """
    Run VADER on a sentence and return the scores.
    """
    doc = nlp(sentence)
    input_to_vader = []

    for sent in doc.sents:
        for token in sent:
            to_add = token.text
            if lemmatize:
                to_add = token.lemma_
                if to_add == '-PRON-': 
                    to_add = token.text
            input_to_vader.append(to_add)

    scores = vader_model.polarity_scores(' '.join(input_to_vader))
    return scores

def vader_output_to_label(vader_output):
    """
    Convert VADER output to a label.
    """
    compound = vader_output['compound']
    if compound >= 0.1:
        return 'positive'
    elif compound <= -0.1:
        return 'negative'
    else:
        return 'neutral'

tqdm.pandas()

def evaluate_vader():
    dataset_combined2 = dataset_combined.sample(frac=0.5, random_state=RANDOM_SEED)
    predictions = dataset_combined2['text'].progress_apply(lambda x: vader_output_to_label(run_vader(x)))
    gold = dataset_combined2['sentiment']

    print("\nEvaluation Results:")
    print(f"Sample prediction: {predictions.iloc[2]}, gold: {gold.iloc[2]}")
    print("\nClassification Report:")
    print(classification_report(gold, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(gold, predictions))

def evaluate_vu_test_data():
    print("\nEvaluating VU_test_data with VADER:\n")
    VU_test_data['vader_score'] = VU_test_data['sentence'].apply(lambda x: vader_output_to_label(run_vader(x)))
    
    for i, row in VU_test_data.iterrows():
        print(f"{i+1}. sentence: {row['sentence']}")
        print(f"   Gold: {row['label']}")
        print(f"   VADER: {row['vader_score']}\n")


if __name__ == "__main__":
    print("\nUsing VADER Sentiment Analysis:")
    evaluate_vader()
    #evaluate_vu_test_data()