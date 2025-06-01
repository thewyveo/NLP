from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import nltk
import pandas as pd


def initialize_vader():
    '''
    Defines (and downloads if necessary) the VADER sentiment analysis model
    and spaCy NLP model. Initializes the part-of-speech tags as an empty set.

    :return: nlp: spaCy NLP model
    :return: vader_model: pre-trained VADER model
    :return: pos: set of part-of-speech tags
    '''

    nltk.download('vader_lexicon')
    nlp = spacy.load("en_core_web_sm")
    vader_model = SentimentIntensityAnalyzer()
    pos = set()
    return nlp, vader_model, pos

def import_test():
    '''
    Imports the test data and gets rid of the first line (headers).

    :return: test_data: pandas DataFrame containing the test data
    '''

    test_data = pd.read_csv('sentiment/datasets/sentiment-topic-test.tsv', sep='\t')
    test_data.drop('sentence_id', axis=1, inplace=True)
    return test_data

def preprocess_train_data():
    '''
    Preprocesses the training data.

    :return: train_data: pandas DataFrame containing the (preprocessed) training data
    '''
    
    train_data = pd.read_csv('sentiment/datasets/processed_sentiment_training.csv', sep=',')
    return train_data