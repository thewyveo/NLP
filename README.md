# NLP
Natural Language Processing Project for the Text Mining Course

## TOPIC MODELING COMPONENT

### Datasets:

1) https://www.cs.cornell.edu/people/pabo/movie-review-data/ (polarity dataset v2.0)
2) https://github.com/krystalan/goal
3) https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data
4) https://www.kaggle.com/datasets/manhdodz249/irish-times-dataset-for-topic-model?resource=download

### Directory Formation: make sure the datasets for the topic modeling component look like this:

<code> ```
topic-modeling/
│    └── datasets/
│        ├── archive/... (1st link)
│        ├── goal/data/... (2nd link)
│        ├── review_polarity/txt_sentoken/neg/... (3rd link)
│        ├── review_polarity/txt_sentoken/pos/... (3rd link II)
│        ├── Books_rating.csv (4th link)
│        └── sentiment-topic-test.tsv (test set)
│
├── lda_lsa.py
├── logisticreg.py
├── preprocessing.py
├── topic_modeling.ipynb (MAIN FILE TO RUN ALL)
└── requirements.txt (ONLY the requirements for the topic modeling component)
``` </code>
