import json
import os
import pandas as pd
import random
import spacy
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

def process_Irish_Times(path_data='datasets/archive/new_IrishTimes_test.txt', path_labels='datasets/archive/new_IrishTimes_test_label.txt'):
    all_labels, all_data, irish_data = [], [], []
    with open(path_labels, 'r') as file:
        for line in file.readlines():
            all_labels.append(line.strip())
    with open (path_data, 'r') as file:
        for line in file.readlines():
            all_data.append(line.strip())
        for index, label in enumerate(all_labels):
            if label == 'sport':
                irish_data.append(all_data[index])
    return irish_data

def process_GOAL(path='datasets/goal/data/goal.json'):
    goal_data = []
    with open(path, 'r') as file:
        data = json.load(file)
    for sublist in data:
        subsublist = sublist['data']['commentary']
        for subsubsublist in subsublist:
            text = subsubsublist[1]
            best_sentence = sorted(text.split('.'), key=len, reverse=True)[0]
            goal_data.append(best_sentence)
    return goal_data

def process_Cornell(path):
    iterations = 0
    movie_data = []
    for filename in os.listdir(path):
        iterations += 1
        if filename.endswith('.txt'):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r') as file:
                file_data = []
                for line in file.readlines():
                    if len(line) < 25:
                        continue
                    file_data.append(line.strip())
                movie_data.append(file_data)
        if iterations > 175:
            break

    movie_data = [item for sublist in movie_data for item in sublist]
    return movie_data

def process_Amazon(path='datasets/Books_rating.csv'):
    book_data = []
    used_books = []
    first_1000 = pd.read_csv(path, nrows=160000)
    first_1000_list = first_1000.values.tolist()
    for sublist in first_1000_list:
        title = sublist[1]
        if title not in used_books:
            used_books.append(title)
            full_text = sublist[-1]
            full_text = full_text.split('.')
            random_idx = random.randint(a=0, b=len(full_text)-1)
            random_sentence = full_text[random_idx]
            #best_sentence = sorted(full_text, key=len, reverse=True)[0]
            book_data.append(random_sentence)
            #book_data.append(best_sentence)
    return book_data

def add_labels(dataset, label):
    labeled_dataset = []
    for instance in dataset:
        labeled = (instance, label)
        labeled_dataset.append(labeled)
    return labeled_dataset

def combine_datasets():
    irish_data = process_Irish_Times()
    goal_data = process_GOAL()
    pos_movie_data = process_Cornell('datasets/review_polarity/txt_sentoken/pos')
    neg_movie_data = process_Cornell('datasets/review_polarity/txt_sentoken/neg')
    sport_data = irish_data + goal_data
    movie_data = pos_movie_data + neg_movie_data
    book_data = process_Amazon()

    labeled_sports_data = add_labels(sport_data, 'sports')
    labeled_movie_data = add_labels(movie_data, 'movie')
    labeled_book_data = add_labels(book_data, 'book')

    final_dataset = labeled_sports_data + labeled_movie_data + labeled_book_data
    return final_dataset

def preprocess():
    dataset = combine_datasets()
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    stop_words = set(stopwords.words('english'))
    processed = []
    for (text, label) in dataset:
        tokens = simple_preprocess(text, deacc=True)
        tokens = [token for token in tokens if token not in stop_words]
        doc_out = nlp(' '.join(tokens))
        lemmatized = [token.lemma_ for token in doc_out if token.lemma_ not in stop_words and len(token.lemma_) > 2]
        processed.append((lemmatized, label))

    processed = [(' '.join(text) if isinstance(text, list) else text, label ) for text, label in processed]

    return processed
