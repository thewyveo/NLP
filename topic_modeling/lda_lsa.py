from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer    # C.V. for LDA, TF-IDF for LSA
from sklearn.linear_model import LogisticRegression     # for both
from sklearn.decomposition import TruncatedSVD      # for lsa
from sklearn.decomposition import LatentDirichletAllocation as LDA  # for lda (duh)
import numpy as np
from sklearn.metrics import classification_report

from sklearn.model_selection import StratifiedKFold   # for preprocessing
from sklearn.preprocessing import LabelEncoder


def lda(dataset, n_components=10):
    '''
    Performs supervised topic modeling using LDA combined with logistic regression to classify text data.
    1. Preprocesses text data and converts labels to numerical values (0, 1, 2) via uses CountVectorizer (BoW approach)
    2. Splits the dataset into training and validation sets with stratified 5-fold cross-validation
    3. Applies LDA to extract (latent) topic distributions from the text (with 10 components)
    4. Fits the topics to a logistic regression classifier
    5. Calculates the mean accuracy across folds
    6. Defines a predict function to classify new text based on the trained model, and tests it on the test (assignment) dataset.
    The key idea behind 10 components for LDA instead of forcing the topics to align directly with the 3 target labels, is that
    this approach allows for richer topic representations. The model learns 10 distinct topics from the text data, which can capture
    a wider range of semantic patterns than it would with just 3. The logistic regression classifier then maps these 10
    topic distributions to the 3 labels, taking the topic distributions as features.
    
    :input: dataset: A list of tuples where the first element is the text and the second element is the label.
    :input: n_components: The number of (latent) topics to extract with LDA (default is 10).
    :return: A tuple in the format ((mean accuracy during training, accuracies across folds history), (predicted topics, true labels), test text, classification report of training).
    '''

    texts = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]

    le = LabelEncoder()     # define a label encoder to convert labels to numerical
    y = le.fit_transform(labels)    # fit the encoder to the labels

    vectorizer = CountVectorizer(stop_words='english', max_features=5000) # co-occurrance matrix with 5k features and stop words removed
    X = vectorizer.fit_transform(texts)

    # the difference between normal kfold and stratified kfold is that stratified kfold ensures
    # each fold has the same proportion of classes as the dataset
    # (in our case, all three classes make up roughly 1/3 of the dataset and are approx. equal)
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    all_preds, all_labels, fold_accuracies = [], [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):       # iterating over folds
        print(f'Epoch (Fold) {fold}/5')      # lda takes a little while so we print the fold number
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx] # data splitting

        lda = LDA(n_components=n_components)      # 10 topic distributions LDA model
        X_train_topics = lda.fit_transform(X_train) # fit LDA on training data and transform to topic distributions
        X_test_topics = lda.transform(X_test)   # apply the same transformation to the test data

        clf = LogisticRegression(max_iter=1000)     # defining,
        clf.fit(X_train_topics, y_train)    # and fitting a logistic regression classifier to the topics

        y_pred = clf.predict(X_test_topics)   # predicting the labels for the validation set
        all_preds.extend(y_pred)        # storing
        all_labels.extend(y_test)
        fold_accuracies.append(clf.score(X_test_topics, y_test))    # track history of accuracies across folds
        # obviously no backpropagation since we are just fitting the model to a co-occurrence matrix

    lda_mean_acc = np.mean(fold_accuracies)
    training_report = classification_report(all_labels, all_preds, 
                                          target_names=le.classes_, 
                                          output_dict=False)

    def predict(x):
        '''
        Predict method for the LDA+Logistic Regression model.

        :input: x: The input text to classify.
        :return: The predicted label for the input text, inverted back to the original label
            (from 0, 1, 2 back to 'sports', 'movie', 'book').

        '''
        vec = vectorizer.transform([x]) # transform input to word count vector using trained vectorizer
        topic_dist = lda.transform(vec) # apply trained LDA model to get topic distribution
        pred_encoded = clf.predict(topic_dist)[0]   # predict the label using the logistic regression classifier
        return le.inverse_transform([pred_encoded])[0] # invert back to original label and return prediction
    
    with open("topic_modeling/datasets/sentiment-topic-test.tsv", "r") as f:
        content = f.readlines()
        test_sentences = [line.split('\t')[1].strip() for line in content][1:] # split by tabs, strip whitespace, skip header line ([1:]).
        true_labels = [line.split('\t')[-1].strip() for line in content][1:] # test sentences are the second col. and true labels the last.
        predicted_topics = [predict(s) for s in test_sentences]
        # predict for each sentence
    
    return (lda_mean_acc, fold_accuracies), (predicted_topics, true_labels), test_sentences, training_report


def lsa(dataset, n_components=10):
    '''
    Performs supervised topic modeling using LSA (Latent Semantic Analysis) combined with logistic regression.
    1. Preprocesses text data and converts labels to numerical values (0, 1, 2), uses TF-IDF for feature extraction
    2. Splits the dataset into training and validation sets with stratified 5-fold cross-validation
    3. Applies TruncatedSVD to extract latent semantic features from the TF-IDF matrix (key LSA step)
    4. Fits the reduced features to a logistic regression classifier
    5. Calculates the mean accuracy across folds
    6. Defines a predict function to classify new text based on the trained model, and tests it on the test dataset.
    The n_components parameter being set to 10 allows the model to learn 10 latent dimensions from the TF-IDF matrix,
    (although not much difference in performance was observed with 3 or 5).
    
    :input: dataset: A list of tuples where the first element is the text and the second element is the label.
    :input: n_components: The number of latent dimensions to extract from the TF-IDF matrix with LSA (default is 10).
    :return: A tuple in the format ((mean accuracy during training, accuracies across folds history), (predicted topics, true labels), test text, classification report of training).
    '''

    texts = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]

    le = LabelEncoder()   # define a label encoder to convert labels to numerical
    y = le.fit_transform(labels)    # fit the encoder to the labels

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # TF-IDF matrix with 5k features and stop words removed
    X = vectorizer.fit_transform(texts)

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    all_preds, all_labels, fold_accuracies = [], [], []
    for fold, (train_idx, test_idx) in (enumerate(skf.split(X, y), 1)):  # iterating over folds
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svd = TruncatedSVD(n_components=n_components)  # define a TruncatedSVD model for LSA with n_components (10)
        X_train_lsa = svd.fit_transform(X_train)    # reduce training data dimensionality
        X_test_lsa = svd.transform(X_test)  # apply the same transformation to the test data

        clf = LogisticRegression(max_iter=1000)   # define & fit a logistic regression classifier
        clf.fit(X_train_lsa, y_train) 

        y_pred = clf.predict(X_test_lsa)    # predict the labels for the validation set
        all_preds.extend(y_pred)    # storing
        all_labels.extend(y_test)
        fold_accuracies.append(clf.score(X_test_lsa, y_test))   # track history of accuracies across folds

    lsa_mean_acc = np.mean(fold_accuracies)
    training_report = classification_report(all_labels, all_preds, 
                                          target_names=le.classes_, 
                                          output_dict=False)

    def predict(x):
        '''
        Predict method for the LSA+Logistic Regression model.

        :input: x: The input text to classify.
        :return: The predicted label for the input text, inverted back to the original label
            (from 0, 1, 2 back to 'sports', 'movie', 'book').
        '''
        vec = vectorizer.transform([x]) # convert text to TF-IDF vector
        topic_dist = svd.transform(vec) # project into LSA space
        pred_encoded = clf.predict(topic_dist)[0]   # predict via log. reg.
        return le.inverse_transform([pred_encoded])[0]  # invert back to original label and return prediction
    
    with open("topic_modeling/datasets/sentiment-topic-test.tsv", "r") as f:       # evaluating on test dataset
        content = f.readlines()
        test_sentences = [line.split('\t')[1].strip() for line in content if line.strip()][1:]
        true_labels = [line.split('\t')[-1].strip() for line in content][1:]
        predicted_topics = [predict(s) for s in test_sentences]

    return (lsa_mean_acc, fold_accuracies), (predicted_topics, true_labels), test_sentences, training_report