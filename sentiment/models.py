import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import nltk
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from pyfiglet import figlet_format


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

def logreg(train_data, test_data): 
    vectorizer = TfidfVectorizer(min_df=2) # Changing min_df doesnt make difference for this dataset

    X_train, X_test, y_train, y_test = train_test_split(
        train_data['text'], 
        train_data['sentiment'], 
        test_size=0.2)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    def predict(x):
        '''
        Predict method for the Logistic Regression model.
        
        :input: x: The input text to classify.
        :return: str: The predicted label for the input text.
        '''
        vec = vectorizer.transform([x]) # convert text to TF-IDF features
        return clf.predict(vec)[0]  # return predicted label

    test_sentences = [line for line in test_data['sentence']]
    true_labels = [line for line in test_data['sentiment']]
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
    train_data_half = train_data.sample(frac=0.2)
    predictions = train_data_half['text'].apply(lambda x: vader_output_to_label(vader(x, nlp, vader_model, pos)))
    gold = train_data_half['sentiment']

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

    print(f"\n=== ST - {model} Results ===")
    print(classification_report(y_true, y_pred))
    if model == 'VADER':
        print(f"\t=== ST - Confusion Matrices ===")
    cm = confusion_matrix(y_true, y_pred, labels=['positive', 'negative', 'neutral'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['positive', 'negative', 'neutral'])
    ax = disp.plot(cmap=plt.cm.Blues)
    ax.ax_.set_title(f'{model} Model')
    plt.show()

def compare_models(logreg_preds, vader_preds, labels, sentences):
    print("\n=== ST - Sentences ===")
    for i, (lr_pred, vader_pred, true_label) in enumerate(zip(logreg_preds, vader_preds, labels)):
        if lr_pred != vader_pred:
            if i != 0:
                print()
            print(f"- Sentence: {sentences[i]}")
            print(f"  True: {true_label} | LogReg: {lr_pred} | VADER: {vader_pred}")
            print()    

    vader_accuracy = float(accuracy_score(labels, vader_preds))
    logreg_accuracy = float(accuracy_score(labels, logreg_preds))
    
    # plotting
    plt.figure(figsize=(16, 8))    # set the figure size
    plt.subplot(1,2,1)    # 1 row, 2 columns, first subplot.
    bars = plt.bar(['VADER', 'Logistic Regression'], [vader_accuracy, logreg_accuracy], color=['red', 'blue'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')

    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(['VADER', 'Logistic Regression'], loc='upper left')
    plt.show()

def Sentiment_Analysis_Component():
    '''
    Sentiment Analysis Component for NLP Pipeline.
    
    This function initializes the training dataset and runs the sentiment analysis models.
    '''
    print()
    print(figlet_format("Sentiment Analysis", font="small"))
    
    # Initialize VADER
    test_data = import_test()
    train_data = preprocess_train_data()
    nlp, vader_model, pos = initialize_vader()
    
    # Run Multinomial Naive Bayes model
    print('\nST - Running Logistic Regression model...')
    (y_pred_logreg, y_true_logreg), test_sentences = logreg(train_data, test_data)
    
    # Run VADER model
    print('ST - Running VADER model (might take a while)...')
    run_vader(train_data, nlp, vader_model, pos)

    y_pred_vader = vader_predict(test_sentences, nlp, vader_model, pos)
    compare_models(y_pred_logreg, y_pred_vader, test_data['sentiment'], test_sentences)
    analyze_results((y_pred_logreg, y_true_logreg), "Logistic Regression")
    analyze_results((y_pred_vader, test_data['sentiment']), "VADER")


#Sentiment_Analysis_Component()