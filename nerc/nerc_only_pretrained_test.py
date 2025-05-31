import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    TrainingArguments, DataCollatorForTokenClassification
)
from transformers import pipeline
import torch
from sklearn.metrics import classification_report
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import spacy
from spacy.training import Example
import random


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
        print("spaCy model loaded successfully")
    except OSError:
        print("spaCy model not found. Please install with: python -m spacy download en_core_web_trf")

    bert_ner = pipeline("ner", 
                model="dslim/bert-base-NER", 
                tokenizer="dslim/bert-base-NER",
                aggregation_strategy="simple",
                framework="pt"
                )
    print("BERT NERC pipeline loaded successfully")

    return nlp_spacy, bert_ner

def models(nlp_spacy, bert_ner):
    sentences, test_data_structured = load_test_data()
    def apply_spacy_ner(sentences):
        """Apply spaCy NER to sentences and return BIO tags"""
        results = []
        
        for sentence in sentences:
            doc =nlp_spacy(sentence)
            tokens = [token.text for token in doc]
            ner_tags = []
            
            for token in doc:
                if token.ent_type_:
                    if token.ent_iob_ == 'B':
                        ner_tags.append(f"B-{token.ent_type_}")
                    elif token.ent_iob_ == 'I':
                        ner_tags.append(f"I-{token.ent_type_}")
                    else:
                        ner_tags.append('O')
                else:
                    ner_tags.append('O')
            
            results.append({'tokens': tokens, 'ner_tags': ner_tags})
        
        return results

    def apply_bert_ner(sentences):
        """Apply BERT NER to sentences and return BIO tags"""
        results = []
        
        for sentence in sentences:
            # Get BERT predictions
            entities = bert_ner(sentence)
            
            # Simple tokenization for alignment
            tokens = sentence.split()
            ner_tags = ['O'] * len(tokens)
            
            # Map BERT entities to tokens
            for entity in entities:
                entity_text = entity['word'].replace('##', '')
                entity_label = entity['entity_group']
                
                # Find matching tokens
                for i, token in enumerate(tokens):
                    if (entity_text.lower() in token.lower() or 
                        token.lower() in entity_text.lower()):
                        if ner_tags[i] == 'O':
                            ner_tags[i] = f"B-{entity_label}"
                        break
            
            results.append({'tokens': tokens, 'ner_tags': ner_tags})
        
        return results

    # Apply both systems to test data
    print("\nApplying spaCy NERC...")
    spacy_results = apply_spacy_ner(sentences)
    print("Applying BERT NERC...")
    bert_results = apply_bert_ner(sentences)

    return test_data_structured, spacy_results, bert_results

def create_comparison_dataframe(test_data_structured, spacy_results, bert_results):
    """Create detailed comparison dataframe"""
    comparison_data = []
    
    for i, (test_data, spacy_result, bert_result) in enumerate(zip(test_data_structured, spacy_results, bert_results)):
        sentence_id = test_data['sentence_id']
        true_tokens = test_data['tokens']
        true_labels = test_data['ner_tags']
        
        spacy_tokens = spacy_result['tokens']
        spacy_labels = spacy_result['ner_tags']
        
        bert_tokens = bert_result['tokens']
        bert_labels = bert_result['ner_tags']
        
        # Align tokens (use true tokens as reference)
        for j, token in enumerate(true_tokens):
            true_label = true_labels[j] if j < len(true_labels) else 'O'
            spacy_label = spacy_labels[j] if j < len(spacy_labels) else 'O'
            bert_label = bert_labels[j] if j < len(bert_labels) else 'O'
            
            comparison_data.append({
                'sentence_id': sentence_id,
                'token': token,
                'true_label': true_label,
                'spacy_pred': spacy_label,
                'bert_pred': bert_label,
                'spacy_correct': spacy_label == true_label,
                'bert_correct': bert_label == true_label,
                'systems_agree': spacy_label == bert_label
            })
    
    return pd.DataFrame(comparison_data)

def compare(test_data_structured, spacy_results, bert_results):
    # Create comparison dataframe
    comparison_df = create_comparison_dataframe(test_data_structured, spacy_results, bert_results)
    return comparison_df

def extract_entities_from_bio(tokens, bio_tags):
    """Extract named entities from BIO-tagged tokens"""
    entities = []
    current_entity = None
    current_tokens = []
    
    for token, tag in zip(tokens, bio_tags):
        if tag.startswith('B-'):
            # Save previous entity if exists
            if current_entity and current_tokens:
                entities.append({
                    'text': ' '.join(current_tokens),
                    'label': current_entity,
                    'tokens': current_tokens.copy()
                })
            
            # Start new entity
            current_entity = tag[2:]  # Remove 'B-' prefix
            current_tokens = [token]
            
        elif tag.startswith('I-') and current_entity == tag[2:]:
            # Continue current entity
            current_tokens.append(token)
            
        else:
            # End current entity
            if current_entity and current_tokens:
                entities.append({
                    'text': ' '.join(current_tokens),
                    'label': current_entity,
                    'tokens': current_tokens.copy()
                })
            current_entity = None
            current_tokens = []
    
    # Don't forget last entity
    if current_entity and current_tokens:
        entities.append({
            'text': ' '.join(current_tokens),
            'label': current_entity,
            'tokens': current_tokens.copy()
        })
    
    return entities

def extract(test_data_structured, spacy_results, bert_results):
    # Extract entities for each system
    print("\nNAMED ENTITY EXTRACTION RESULTS")
    print("="*60)

    for i, (test_data, spacy_result, bert_result) in enumerate(zip(test_data_structured, spacy_results, bert_results)):
        sentence = test_data['sentence']
        true_entities = extract_entities_from_bio(test_data['tokens'], test_data['ner_tags'])
        spacy_entities = extract_entities_from_bio(spacy_result['tokens'], spacy_result['ner_tags'])
        bert_entities = extract_entities_from_bio(bert_result['tokens'], bert_result['ner_tags'])
        
        print(f"\nSentence {test_data['sentence_id']+1}: {sentence}")
        print(f"True entities: {[(e['text'], e['label']) for e in true_entities]}")
        print(f"spaCy entities: {[(e['text'], e['label']) for e in spacy_entities]}")
        print(f"BERT entities: {[(e['text'], e['label']) for e in bert_entities]}")

def analysis(test_data_structured, spacy_results, bert_results):
        # Detailed error analysis
    print("\nFINAL ANALYSIS & ERROR PATTERNS")
    print("=" * 60)

    # Overall performance
    comparison_df = compare(test_data_structured, spacy_results, bert_results)
    print(f"Total tokens analyzed: {len(comparison_df)}")
    print(f"spaCy accuracy: {comparison_df['spacy_correct'].mean():.4f}")
    print(f"BERT accuracy: {comparison_df['bert_correct'].mean():.4f}")
    print(f"System agreement: {comparison_df['systems_agree'].mean():.4f}")
    spacy_accuracy = comparison_df['spacy_correct'].mean()
    bert_accuracy = comparison_df['bert_correct'].mean()

    # Error patterns
    spacy_errors = comparison_df[~comparison_df['spacy_correct']]
    bert_errors = comparison_df[~comparison_df['bert_correct']]

    print(f"\nError Summary:")
    print(f"spaCy errors: {len(spacy_errors)} tokens")
    print(f"BERT errors: {len(bert_errors)} tokens")

    if len(spacy_errors) > 0:
        print("\nMost common spaCy error patterns:")
        spacy_error_patterns = spacy_errors.groupby(['true_label', 'spacy_pred']).size().reset_index(name='count')
        print(spacy_error_patterns.sort_values('count', ascending=False).head())

    if len(bert_errors) > 0:
        print("\nMost common BERT error patterns:")
        bert_error_patterns = bert_errors.groupby(['true_label', 'bert_pred']).size().reset_index(name='count')
        print(bert_error_patterns.sort_values('count', ascending=False).head())


def NERC_Component():
    print("Named Entity Recognition Component:\n")
    # Load test data
    sentences, test_data_structured = load_test_data()
    
    # Load pretrained models
    nlp_spacy, bert_ner = load_pretrained_models()
    
    # Apply models
    test_data_structured, spacy_results, bert_results = models(nlp_spacy, bert_ner)
    
    # Compare results
    compare(test_data_structured, spacy_results, bert_results)
    
    # Extract entities
    extract(test_data_structured, spacy_results, bert_results)
    
    # Final analysis
    analysis(test_data_structured, spacy_results, bert_results)