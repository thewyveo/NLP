# NLP
Natural Language Processing Project for the Text Mining Course.


## REQUIREMENTS
Please download the requirements from requirements.txt with the following command:
<code>
pip install -r requirements.txt
</code>


## HOW TO RUN
Once the directory is formed like shown below, please **run the app.ipynb file**
    (the main script to integrate and run all components).


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

1) Y. Hou et al., “Bridging Language and Items for Retrieval and Recommendation.” Available: https://arxiv.org/pdf/2403.03952
2) R. Socher et al., “Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank,” Association for Computational Linguistics, 2013. Available: https://aclanthology.org/D13-1170.p

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

1) B. Pang and L. Lee, "Movie Review Data," Cornell University, 2005. [Online]. Available: https://www.cs.cornell.edu/people/pabo/movie-review-data/.
2) M. Bakhet, "Amazon Books Reviews Dataset," Kaggle, 2018. [Online]. Available: https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data.
3) K. Li et al., "GOAL: A Dataset for Goal-Oriented Dialogues," GitHub, 2022. [Online]. Available: https://github.com/krystalan/goal. 
4) M. Dodz, "Irish Times Dataset for Topic Modeling," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/manhdodz249/irish-times-dataset-for-topic-model?resource=download.

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
