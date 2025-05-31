from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import nltk
import pandas as pd


def initialize_vader():
    nltk.download('vader_lexicon')
    nlp = spacy.load("en_core_web_sm")
    vader_model = SentimentIntensityAnalyzer()
    pos = set()
    return nlp, vader_model, pos

def import_test():
    test_data = pd.read_csv('sentiment/datasets/sentiment-topic-test.tsv', sep='\t')
    test_data.drop('sentence_id', axis=1, inplace=True)
    return test_data

def preprocess_train_data():
    train_data = pd.read_csv('sentiment/datasets/processed_sentiment_training.csv', sep=',')
    return train_data