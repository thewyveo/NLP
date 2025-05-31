import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import nltk
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def initialize_vader():
    nltk.download('vader_lexicon')
    nlp = spacy.load("en_core_web_sm")
    vader_model = SentimentIntensityAnalyzer()
    pos = set()
    return nlp, vader_model, pos

def import_test():
    test_data = pd.read_csv('sentiment-topic-test.tsv', sep='\t')
    test_data.drop('sentence_id', axis=1, inplace=True)
    return test_data

def preprocess_train_data():
    sentences = pd.read_csv('stanfordSentimentTreebank/datasetSentences.txt', sep='\t')
    sentiments = pd.read_csv('stanfordSentimentTreebank/sentiment_labels.txt', sep='|', engine='python')
    sentiments.columns = ['sentence_index', 'sentiment_value']

    train_data_stf = pd.merge(sentences, sentiments, on='sentence_index')
    train_data_stf = train_data_stf.rename(columns={'sentence': 'text', 'sentiment_value': 'rating'})
    train_data_stf.drop('sentence_index', axis=1, inplace=True)
    
    data_amazon = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023", 
    "raw_review_Books", 
    split="full[:100000]"
    )

    df = pd.DataFrame(data_amazon)
    train_data_amazon = df[['rating', 'text']]

    min_count = train_data_amazon['rating'].value_counts().min()
    train_data_amazon = train_data_amazon.groupby('rating').apply(
    lambda x: x.sample(n=min_count, random_state=42)
    ).reset_index(drop=True)

    train_data_stf['rating'] = (train_data_stf['rating'] * 5).clip(1, 5).round()
    train_data_combined = pd.concat([train_data_amazon, train_data_stf], ignore_index=True)
    train_data_combined = train_data_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return train_data_combined

def rating_to_sentiment(rating):
    if rating >= 3.0:
        return 'positive'
    elif rating <= 1.0:
        return 'negative'
    else:
        return 'neutral'

def logreg(train_data, test_data): 
    vectorizer = TfidfVectorizer(min_df=2) # Changing min_df doesnt make difference for this dataset

    train_data['sentiment'] = train_data['rating'].apply(rating_to_sentiment)

    X_train, X_test, y_train, y_test = train_test_split(
        train_data['text'], 
        train_data['sentiment'], 
        test_size=0.2, 
        random_state=42)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    def predict(x):
        '''
        Predict method for the Logistic Regression model.
        
        :input: x: The input text to classify.
        :return: str: The predicted label for the input text.
        '''
        vec = vectorizer.transform([x]) # convert text to TF-IDF features
        return clf.predict(vec)[0]  # return predicted label

    print(test_data['sentence'].head())  # print the first few lines of the test data
    test_sentences = [line for line in test_data['sentence']]  # 
    true_labels = [line for line in test_data['sentiment']]    # 
    predicted_topics = [predict(s) for s in test_sentences]
    # predict for each sentence
    
    return (predicted_topics, true_labels), test_sentences

def vader(sentence, nlp, vader_model, pos, lemmatize=True):
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

            if pos:
                if token.pos_ in pos:
                    input_to_vader.append(to_add) 
            else:
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
    
def run_vader(train_data, nlp, vader_model, pos):
    train_data_half = train_data.sample(frac=0.1, random_state=42)
    predictions = train_data_half['text'].apply(lambda x: vader_output_to_label(vader(x, nlp, vader_model, pos)))
    gold = train_data_half['sentiment']

    print(f"Sample prediction: {predictions.iloc[2]}, gold: {gold.iloc[2]}")
    print(classification_report(gold, predictions))
    print(confusion_matrix(gold, predictions))
    print(f'accuracy: {accuracy_score(gold, predictions)}')

def vader_predict(sentences, nlp, vader_model, pos):
    """
    Predict sentiment labels for a list of sentences using VADER.
    """
    predictions = []
    for sentence in sentences:
        scores = vader(sentence, nlp, vader_model, pos)
        label = vader_output_to_label(scores)
        predictions.append(label)
    return predictions

def analyze_results(results, model):
    """
    Compare predictions from two sentiment models.
    
    :param results: Tuple containing (logreg_preds, gold_labels), (vader_preds, gold_labels)
    """

    (y_pred, y_true) = results if model != 'VADER' else results
    test_accuracy = accuracy_score(y_true, y_pred)

    print(f"\n=== Results of {model} ===")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=['positive', 'negative', 'neutral']))
    print(f"Accuracy: {test_accuracy:.2f}")

def compare_models(logreg_preds, vader_preds, labels, sentences):
    for i, (lr_pred, vader_pred, true_label) in enumerate(zip(logreg_preds, vader_preds, labels)):
        if lr_pred != vader_pred:
            print(f"- Sentence: {sentences[i]}")
            print(f"  True: {true_label} | LogReg: {lr_pred} | VADER: {vader_pred}")
            print()    

    vader_accuracy = accuracy_score(labels, vader_preds)
    logreg_accuracy = accuracy_score(labels, logreg_preds)
    
    # plotting
    plt.figure(figsize=(16, 8))    # set the figure size
    plt.subplot(1,2,1)    # 1 row, 2 columns, first subplot.
    plt.axhline(y=vader_accuracy, color='red', linestyle='--', label='VADER Accuracy')
    plt.axhline(y=logreg_accuracy, color='blue', linestyle=':', label='Logistic Regression Accuracy')
    plt.title('Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.ylim(0.4, 1)    # scope of the 'y' axis
    plt.legend()    # displays what each line represents
    plt.show()


def Sentiment_Analysis_Component():
    '''
    Sentiment Analysis Component for NLP Pipeline.
    
    This function initializes the training dataset and runs the sentiment analysis models.
    '''
    print('\n\nSentiment Analysis Component:\n')
    
    # Initialize VADER
    test_data = import_test()
    train_data = preprocess_train_data()
    nlp, vader_model, pos = initialize_vader()
    
    # Run Multinomial Naive Bayes model
    print('Running Logistic Regression model...')
    (y_pred_logreg, y_true_logreg), test_sentences = logreg(train_data, test_data)
    
    # Run VADER model
    print('Running VADER model...')
    run_vader(train_data, nlp, vader_model, pos)

    y_pred_vader = vader_predict(test_sentences, nlp, vader_model, pos)
    analyze_results((y_pred_logreg, y_true_logreg), "Logistic Regression")
    analyze_results((y_pred_vader, test_data['sentiment']), "VADER")
    compare_models(y_pred_logreg, y_pred_vader, test_data['sentiment'], test_sentences)


if __name__ == "__main__":
    Sentiment_Analysis_Component()