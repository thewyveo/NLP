from sklearn.feature_extraction.text import TfidfVectorizer     # model
from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.model_selection import StratifiedKFold     # preprocessing


def logreg(dataset):
    '''
    Performs text classification using Logistic Regression with TF-IDF features.
    1. Converts raw text into TF-IDF feature vectors.
    2. Uses stratified 5-fold cross-validation to evaluate performance.
    3. Trains (fits) a logistic regression classifier on the text features.
    4. Evaluates on test data and provides prediction capability for new text.
    
    :input: dataset: A list of tuples where each element is a tuple of (text, label).
    :return: A tuple in the format ((mean accuracy during training, accuracies across folds history), (predicted topics, true labels), test text).
    '''

    texts = [x[0] for x in dataset]     # extracting texts from the dataset
    labels = [x[1] for x in dataset]    # extract labels

    skf = StratifiedKFold(n_splits=5, shuffle=True)     # stratified 5-fold cross-validation
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)  # TF-IDF vectorizer with 10k features and stop words removed
    X = vectorizer.fit_transform(texts)   # fit the vectorizer to the texts
    y = np.array(labels)    # convert labels to numpy array for compatibility

    all_preds = []
    all_labels = []
    fold_accuracies = []
    for train_index, test_index in skf.split(X, y):     # iterating over the folds
        X_train_vec, X_test_vec = X[train_index], X[test_index]     # split the data into training and test sets
        y_train, y_test = y[train_index], y[test_index]     # split labels accordingly

        clf = LogisticRegression(max_iter=1000)  # define log. reg. model with a maximum of 1000 iterations
        clf.fit(X_train_vec, y_train)   # fit the model to the training data

        y_pred = clf.predict(X_test_vec)   # predict the labels for the test set
        all_preds.extend(y_pred)    # store predictions
        all_labels.extend(y_test)
        fold_accuracies.append(clf.score(X_test_vec, y_test))   # store fold history
    
    training_mean_acc = np.mean(fold_accuracies)

    def predict(x):
        '''
        Predict method for the Logistic Regression model.
        
        :input: x: The input text to classify.
        :return: str: The predicted label for the input text.
        '''
        vec = vectorizer.transform([x]) # convert text to TF-IDF features
        return clf.predict(vec)[0]  # return predicted label

    with open("topic_modeling/datasets/sentiment-topic-test.tsv", "r") as f:
        content = f.readlines()
        test_sentences = [line.split('\t')[1].strip() for line in content][1:]  # split by tabs, strip whitespace, skip header line ([1:]).
        true_labels = [line.split('\t')[-1].strip() for line in content][1:]    # test sentences are the second col. and true labels the last.
        predicted_topics = [predict(s) for s in test_sentences]
        # predict for each sentence

    return (training_mean_acc, fold_accuracies), (predicted_topics, true_labels), test_sentences