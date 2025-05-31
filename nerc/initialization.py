from transformers import pipeline
import spacy
import pandas as pd


def load_test_data():
    """Load and process the NER-test.tsv file"""
    test_df = pd.read_csv('nerc/datasets/NER-test.tsv', sep='\t')
    
    # Group by sentence_id to reconstruct sentences
    sentences = []
    sentence_data = []
    
    for sentence_id in test_df['sentence_id'].unique():
        sentence_tokens = test_df[test_df['sentence_id'] == sentence_id]
        
        tokens = sentence_tokens['token'].tolist()
        ner_tags = sentence_tokens['BIO_NER_tag'].tolist()
        sentence_text = ' '.join(tokens)
        
        sentences.append(sentence_text)
        sentence_data.append({
            'sentence_id': sentence_id,
            'tokens': tokens,
            'ner_tags': ner_tags,
            'sentence': sentence_text
        })
    
    return sentences, sentence_data
    
def load_pretrained_models():
    try:
        nlp_spacy = spacy.load("en_core_web_trf")
        print("NERC - spaCy model loaded successfully.")
    except OSError:
        print("NERC - spaCy model not found. Please install with: python -m spacy download en_core_web_trf")

    bert_ner = pipeline("ner", 
                model="dslim/bert-base-NER", 
                tokenizer="dslim/bert-base-NER",
                aggregation_strategy="simple",
                framework="pt"
                )
    print("NERC - BERT NERC pipeline loaded successfully")

    return nlp_spacy, bert_ner