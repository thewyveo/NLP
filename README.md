# NLP
Natural Language Processing Project for the Text Mining Course


## REQUIREMENTS
Please download the requirements from requirements.txt with the following command:
<code>
pip install -r requirements.txt
</code>


## NAMED ENTITY RECOGNITION & CLASSIFICATION COMPONENT
### Datasets:
No training datasets for this component.

### Directory Formation:
Make sure the directory for the named entity classification & recognition component looks like this:
<code>
nerc/
│  └── datasets/
│        └── NER-test.tsv (test set)
│
├── analysis.py
├── initialization.py
└── models.ipynb (main named entity recognition & classification script)
</code>


## SENTIMENT ANALYSIS COMPONENT
### Datasets:
These datasets were preprocessed and combined to form the processed_sentiment_training.csv file, via sentiment-data-preprocessing.ipynb:

1) # (FATIH BENI EKLE)
2) # (FATIH BENI EKLE)

### Directory Formation:
Make sure the directory for the sentiment analysis component looks like this:
<code>
sentiment/
│    └── datasets/
│        ├── processed_sentiment_training.csv/... (combined, preprocessed training dataset)
│        └── sentiment-topic-test.tsv (test set)
│
├── analysis.py
├── initialize.py
├── models.ipynb (main sentiment analysis script)
└── sentiment-data-preprocessing.ipynb
</code>


## TOPIC MODELING COMPONENT
### Datasets:
The datasets must be imported locally (due to their size), and will be preprocessed when the component is run.

1) https://www.cs.cornell.edu/people/pabo/movie-review-data/ (polarity dataset v2.0)
2) https://github.com/krystalan/goal
3) https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data
4) https://www.kaggle.com/datasets/manhdodz249/irish-times-dataset-for-topic-model?resource=download

### Directory Formation:
Make sure the directory for the topic modeling component looks like this:
<code>
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
├── models.ipynb (main topic modeling script)
├── preprocessing.py
└── plotting.py
</code>
