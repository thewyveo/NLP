from pyfiglet import figlet_format
from nerc.initialization import load_test_data, load_pretrained_models
from nerc.analysis import compare, extract, analysis

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
    print("NERC - Applying pre-trained spaCy model to test data...")
    spacy_results = apply_spacy_ner(sentences)
    print("NERC - Applying pre-trained BERT model to test data...")
    bert_results = apply_bert_ner(sentences)

    return test_data_structured, spacy_results, bert_results


def NERC_Component():
    print(figlet_format("Named Entity Recognition & Classification", font="small"))
    # Load test data
    _, test_data_structured = load_test_data()
    
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


#NERC_Component()