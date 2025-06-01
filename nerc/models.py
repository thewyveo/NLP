from pyfiglet import figlet_format
from nerc.initialization import load_test_data, load_pretrained_models
from nerc.analysis import compare, extract, analysis

def models(nlp_spacy, bert_ner):
    '''
    Loads the test data, defines functions to apply the NER models, and applies them to the test data
    with BIO tags for each token.

    :input: nlp_spacy: the spaCy NERC model
    :input: bert_ner: the BERT NERC pipeline
    :return: test_data_structured: structured test data with tokens and BIO tags
    :return: spacy_results: results from spaCy NERC
    :return: bert_results: results from BERT NERC
    '''

    sentences, test_data_structured = load_test_data()

    def apply_spacy_ner(sentences):
        '''
        Apply spaCy NERC to sentences and return BIO tags
        :input: sentences: a list of sentences to analyze
        :return: a list of dictionaries with tokens and their corresponding BIO tags
        '''
        results = []
        
        for sentence in sentences:      # for each sentence, get predictions,
            doc =nlp_spacy(sentence)
            tokens = [token.text for token in doc]
            ner_tags = []
            
            for token in doc:       # map each token to the BIO tags,
                if token.ent_type_:     # if the token is part of an entity,
                    if token.ent_iob_ == 'B':       # e.g Beginning,
                        ner_tags.append(f"B-{token.ent_type_}") # store tags in temporary list for each sentence
                    elif token.ent_iob_ == 'I':
                        ner_tags.append(f"I-{token.ent_type_}")
                    else:
                        ner_tags.append('O')
                else:
                    ner_tags.append('O')
            
            results.append({'tokens': tokens, 'ner_tags': ner_tags}) # and store the combined results for each sentence
                                                                       # for each sentence after iterating over tokens
        return results

    def apply_bert_ner(sentences):
        '''
        Apply BERT NERC to sentences and return BIO tags
        :input: sentences: a list of sentences to analyze
        :return: a list of dictionaries with tokens and their corresponding BIO tags
        '''
        results = []
        
        for sentence in sentences:      # for each sentence, get predictions,
            entities = bert_ner(sentence)
            tokens = sentence.split()
            ner_tags = ['O'] * len(tokens)
            
            for entity in entities:     # map each entity to the tokens,
                entity_text = entity['word'].replace('##', '')
                entity_label = entity['entity_group']
                
                for i, token in enumerate(tokens):      # find the tokens that matches the entity,
                    if (entity_text.lower() in token.lower() or 
                        token.lower() in entity_text.lower()):
                        if ner_tags[i] == 'O':
                            ner_tags[i] = f"B-{entity_label}"
                        break
            
            results.append({'tokens': tokens, 'ner_tags': ner_tags})        # and store.
        
        return results

    # apply both systems to test data
    print("NERC - Applying pre-trained spaCy model to test data...")
    spacy_results = apply_spacy_ner(sentences)
    print("NERC - Applying pre-trained BERT model to test data...")
    bert_results = apply_bert_ner(sentences)

    return test_data_structured, spacy_results, bert_results


def Named_Entity_Recognition_Classification_Component():
    '''
    Main function to run the NERC component. Loads the test data,
    loads the pre-trained models, applies the models to the test data,
    compares the models and analyses the results.
    '''
    
    print(figlet_format("Named Entity Recognition & Classification", font="small"))
    _, test_data_structured = load_test_data()  # load test data
    nlp_spacy, bert_ner = load_pretrained_models()  # load pretrained models
    test_data_structured, spacy_results, bert_results = models(nlp_spacy, bert_ner) # apply models to test data
    compare(test_data_structured, spacy_results, bert_results)      # compare (raw) results from both models
    extract(test_data_structured, spacy_results, bert_results)      # extract entities from results
    analysis(test_data_structured, spacy_results, bert_results)     # compute final analysis of results


#Named_Entity_Recognition_Classification_Component()