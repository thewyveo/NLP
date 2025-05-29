from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np


def logreg(dataset):
    texts = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    all_preds = []
    all_labels = []
    fold_accuracies = []
    for train_index, test_index in skf.split(X, y):
        X_train_vec, X_test_vec = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_vec, y_train)

        y_pred = clf.predict(X_test_vec)
        all_preds.extend(y_pred)
        all_labels.extend(y_test)
        fold_accuracies.append(clf.score(X_test_vec, y_test))
    
    training_mean_acc = np.mean(fold_accuracies)

    def predict(x):
        vec = vectorizer.transform([x])
        return clf.predict(vec)[0]

    with open("datasets/sentiment-topic-test.tsv", "r") as f:
        content = f.readlines()
        test_sentences = [line.split('\t')[1].strip() for line in content if line.strip()][1:]
        predicted_topics = [predict(s) for s in test_sentences]
        true_labels = [line.split('\t')[-1].strip() for line in content][1:]

    return training_mean_acc, (predicted_topics, true_labels), test_sentences
