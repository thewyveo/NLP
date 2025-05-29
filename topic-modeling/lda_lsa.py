from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np


def lda(dataset, n_topics=10, n_splits=5):
    texts = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]

    le = LabelEncoder()
    y = le.fit_transform(labels)

    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(texts)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_preds, all_labels, fold_accuracies = [], [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f'{fold}/5')
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        lda = LDA(n_components=n_topics, random_state=42)
        X_train_topics = lda.fit_transform(X_train)
        X_test_topics = lda.transform(X_test)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_topics, y_train)

        y_pred = clf.predict(X_test_topics)
        all_preds.extend(y_pred)
        all_labels.extend(y_test)
        fold_accuracies.append(clf.score(X_test_topics, y_test))

    lda_mean_acc = np.mean(fold_accuracies)

    def predict(x):
        vec = vectorizer.transform([x])
        topic_dist = lda.transform(vec)
        pred_encoded = clf.predict(topic_dist)[0]
        return le.inverse_transform([pred_encoded])[0]
    
    with open("datasets/sentiment-topic-test.tsv", "r") as f:
        content = f.readlines()
        test_sentences = [line.split('\t')[1].strip() for line in content if line.strip()][1:]
        predicted_topics = [predict(s) for s in test_sentences]
        true_labels = [line.split('\t')[-1].strip() for line in content][1:]
    
    return lda_mean_acc, (predicted_topics, true_labels), test_sentences


def lsa(dataset, n_topics=10, n_splits=5):
    texts = [x[0] for x in dataset]
    labels = [x[1] for x in dataset]

    le = LabelEncoder()
    y = le.fit_transform(labels)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(texts)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_preds, all_labels, fold_accuracies = [], [], []
    for fold, (train_idx, test_idx) in (enumerate(skf.split(X, y), 1)):
        print(f'{fold}/5')
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        svd = TruncatedSVD(n_components=n_topics, random_state=42)
        X_train_lsa = svd.fit_transform(X_train)
        X_test_lsa = svd.transform(X_test)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_lsa, y_train)

        y_pred = clf.predict(X_test_lsa)
        all_preds.extend(y_pred)
        all_labels.extend(y_test)
        fold_accuracies.append(clf.score(X_test_lsa, y_test))

    lsa_mean_acc = np.mean(fold_accuracies)

    def predict(x):
        vec = vectorizer.transform([x])
        topic_dist = svd.transform(vec)
        pred_encoded = clf.predict(topic_dist)[0]
        return le.inverse_transform([pred_encoded])[0]
    
    with open("datasets/sentiment-topic-test.tsv", "r") as f:
        content = f.readlines()
        test_sentences = [line.split('\t')[1].strip() for line in content if line.strip()][1:]
        predicted_topics = [predict(s) for s in test_sentences]
        true_labels = [line.split('\t')[-1].strip() for line in content][1:]

    return lsa_mean_acc, (predicted_topics, true_labels), test_sentences
