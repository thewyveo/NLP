import pandas as pd


def compare(test_data_structured, spacy_results, bert_results):
    """Create detailed comparison dataframe"""
    comparison_data = []
    
    for i, (test_data, spacy_result, bert_result) in enumerate(zip(test_data_structured, spacy_results, bert_results)):
        sentence_id = test_data['sentence_id']
        true_tokens = test_data['tokens']
        true_labels = test_data['ner_tags']
        
        spacy_labels = spacy_result['ner_tags']
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
    print("\n=== NERC - Results ===")

    for i, (test_data, spacy_result, bert_result) in enumerate(zip(test_data_structured, spacy_results, bert_results)):
        sentence = test_data['sentence']
        true_entities = extract_entities_from_bio(test_data['tokens'], test_data['ner_tags'])
        spacy_entities = extract_entities_from_bio(spacy_result['tokens'], spacy_result['ner_tags'])
        bert_entities = extract_entities_from_bio(bert_result['tokens'], bert_result['ner_tags'])
        if i != 0:
            print()
        print(f"Sentence {test_data['sentence_id']+1}: {sentence}")
        print(f"True entities: {[(e['text'], e['label']) for e in true_entities]}")
        print(f"spaCy entities: {[(e['text'], e['label']) for e in spacy_entities]}")
        print(f"BERT entities: {[(e['text'], e['label']) for e in bert_entities]}")

def analysis(test_data_structured, spacy_results, bert_results):
    print("\n=== NERC - Final Analysis ====")

    # Overall performance
    comparison_df = compare(test_data_structured, spacy_results, bert_results)
    print(f"Total tokens analyzed: {len(comparison_df)}")
    print(f"spaCy accuracy: {comparison_df['spacy_correct'].mean():.4f}")
    print(f"BERT accuracy: {comparison_df['bert_correct'].mean():.4f}")
    print(f"System agreement: {comparison_df['systems_agree'].mean():.4f}")

    # Error patterns
    spacy_errors = comparison_df[~comparison_df['spacy_correct']]
    bert_errors = comparison_df[~comparison_df['bert_correct']]

    print(f"spaCy errors: {len(spacy_errors)} tokens")
    print(f"BERT errors: {len(bert_errors)} tokens")

    if len(spacy_errors) > 0:
        print("\n=== NERC - Most Common spaCy Error Patterns ===")
        spacy_error_patterns = spacy_errors.groupby(['true_label', 'spacy_pred']).size().reset_index(name='count')
        print(spacy_error_patterns.sort_values('count', ascending=False).head())

    if len(bert_errors) > 0:
        print("\n=== NERC - Most Common BERT Error Patterns ===")
        bert_error_patterns = bert_errors.groupby(['true_label', 'bert_pred']).size().reset_index(name='count')
        print(bert_error_patterns.sort_values('count', ascending=False).head())