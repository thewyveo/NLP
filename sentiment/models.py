from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from pyfiglet import figlet_format
from sentiment.initialize import initialize_vader, import_test, preprocess_train_data
from sentiment.analysis import analyze_results, compare_models
from sklearn.pipeline import Pipeline
import numpy as np


def logreg(train_data, test_data):
    '''
    Logistic Regression model for sentiment analysis using stratified 5-fold cross validation.
    1. Converts raw text into TF-IDF feature vectors.
    2. Uses stratified 5-fold cross-validation to boost performance. (dataset is small,
    but when we implemented CV in the training phase, the accuracy went up)
    3. Trains (fits) a logistic regression classifier on the text features.
    4. Predicts on test data and returns results.

    :input: train_data: the training dataset containing
    :input: test_data: the test dataset
    :return: (predicted topics, true labels): a tuple containing the predicted and the true labels
    :return: test_sentences: a list of sentences from the test dataset
    ''' 

    vectorizer = TfidfVectorizer(min_df=2)      # initialize TF-IDF vectorizer with minimum document frequency of 2
          # meaning that only words appearing in at least 2 documents will be considered (to reduce noise)

    pipeline = Pipeline([       # create a pipeline for the model
        ('vectorizer', vectorizer),     # add TF-IDF vectorizer to the pipeline
        ('classifier', LogisticRegression(max_iter=1000))     # add logistic regression classifier to the pipeline
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True)     # initialize stratified 5-fold cross-validation

    X_train = train_data['text']     # extract texts from training data
    y_train = train_data['sentiment']     # extract labels from training data

    cv_scores = cross_val_score(    # perform cross-validation on the model using the pipeline
        pipeline, 
        X_train, 
        y_train, 
        cv=skf,
        scoring='accuracy'
    )     # stores cv scores for each fold

    training_mean_acc = np.mean(cv_scores)     # calculate mean accuracy across all folds
    print(f"\nST - Logistic Regression Model Training Complete.\nTraining Mean Accuracy: {training_mean_acc:.4f}\n")  # print training mean accuracy
    pipeline.fit(X_train, y_train)      # fit the pipeline on the training data
    
    # test part
    predictions = pipeline.predict(test_data['sentence'])  # predict sentiment labels for the test data
    return (predictions, test_data['sentiment']), test_data['sentence']  # return (predictions, true labels), test sentences

def vader(sentence, nlp, vader_model, pos, lemmatize=True):
    '''
    VADER pre-trained model for sentiment analysis.

    :input: sentence: the input sentence to analyze
    :input: nlp: the spaCy NLP model
    :input: vader_model: the VADER model
    :input: pos: list of part-of-speech tags to filter tokens
    :input: lemmatize: whether to lemmatize the tokens (defaults to True)
    :return: a dictionary of VADER sentiment scores
    '''

    doc = nlp(sentence)
    input_to_vader = []

    for sent in doc.sents:
        for token in sent:
            to_add = token.text

            if lemmatize:       # lemmatize token if possible
                to_add = token.lemma_

                if to_add == '-PRON-':  # handle pronuns
                    to_add = token.text

            if pos:     # filter token by pos tag(s)
                if token.pos_ in pos:
                    input_to_vader.append(to_add) 
            else:
                input_to_vader.append(to_add)

    scores = vader_model.polarity_scores(' '.join(input_to_vader))  # get VADER polarity (positive/negative etc.) scores

    return scores

def vader_output_to_label(vader_output):
    '''
    Converts VADER output to a label.

    :input: vader_output: the output from the VADER model
    :return: the converted sentiment label (either 'positive', 'negative', or 'neutral') in string format
    '''

    compound = vader_output['compound']
    
    if compound >= 0.1:
        return 'positive'
    elif compound <= -0.1:
        return 'negative'
    else:
        return 'neutral'
    
def run_vader(train_data, nlp, vader_model, pos):
    '''
    Main funcion to run the VADER components.
    
    :input: train_data: the training dataset
    :input: nlp: the spaCy NLP model
    :input: vader_model: the VADER model
    :input: pos: list of part-of-speech tags to filter tokens
    '''

    train_data_half = train_data.sample(frac=0.2)
    _ = train_data_half['text'].apply(lambda x: vader_output_to_label(vader(x, nlp, vader_model, pos)))
    ## the 'predictions' here is ignored/unused and turned into '_' as it is the training predictions which we do not need to store. -k

def vader_predict(sentences, nlp, vader_model, pos):
    '''
    Predicts sentiment labels for a list of sentences using the VADER model.

    :input: sentences: a list of sentences to analyze
    :input: nlp: the spaCy NLP model
    :input: vader_model: the VADER model
    :input: pos: list of part-of-speech tags to filter tokens
    :return: a list of predicted sentiment labels for each sentence
    '''

    predictions = []
    for sentence in sentences:
        scores = vader(sentence, nlp, vader_model, pos)
        label = vader_output_to_label(scores)
        predictions.append(label)

    return predictions


def Sentiment_Analysis_Component():
    '''
    Main function to run the Sentiment Analysis component. Loads the test data,
    preprocesses the training data, initializes the VADER model, trains both models on the
    training data, and predicts sentiment labels for the test data for each model. Finally,
    this function compares the results of both models and analyzes the results. (the logistic
    regression model's prediction function is defined within the function, whereas the VADER
    model's prediction function is defined outside and called after the VADER model is run)
    '''

    print()
    print(figlet_format("Sentiment Analysis", font="small"))
    
    test_data = import_test()
    train_data = preprocess_train_data()
    nlp, vader_model, pos = initialize_vader()

    print('\nST - Running Logistic Regression model...')
    (y_pred_logreg, y_true_logreg), test_sentences = logreg(train_data, test_data)

    print('ST - Running VADER model (might take a while)...')
    run_vader(train_data, nlp, vader_model, pos)

    y_pred_vader = vader_predict(test_sentences, nlp, vader_model, pos)
    compare_models(y_pred_logreg, y_pred_vader, test_data['sentiment'], test_sentences)
    analyze_results((y_pred_logreg, y_true_logreg), "Logistic Regression")
    analyze_results((y_pred_vader, test_data['sentiment']), "VADER")


#Sentiment_Analysis_Component()