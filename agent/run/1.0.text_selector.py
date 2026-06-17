# © Copyright European Union - 2026

import json

def text_selector(input_file, translation_file, output_file):
    """
    Preprocess the data by selecting the correct text field based on the language.
    If the language is not English, use the provided translation.
    
    Parameters:
    - input_file (str): Path to the input JSON file containing the documents.
    - translation_file (str): Path to the JSON file containing the translations.
    - output_file (str): Path to the output JSON file where the processed data will be saved.
    """
    # Load the input data
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Load the translated data
    with open(translation_file, 'r', encoding='utf-8') as f:
        translated_documents = json.load(f)
    
    # Create a dictionary for quick lookup of translated text
    translation_dict = {doc['id']: doc.get('text_en', doc['text']) for doc in translated_documents}
    
    # Process each document to select the correct text field
    processed_data = []
    for doc in documents:
        doc_id = doc['id']
        language = doc['language']
        
        if language == 'en':
            text = doc['text']
        else:
            text = translation_dict.get(doc_id, doc['text'])  # Fallback to original text if translation is missing
        
        processed_data.append({
            "id": doc_id,
            "text": text
        })
    
    # Save the processed data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    print(f"Processed data saved to {output_file}")

# Example usage:
# text_selector('../data/proposals_tiny.json',
#               '../data/proposals_tiny_translation.json',
#               'processed_data.json')
```

This function `text_selector` reads the input JSON file, checks the language of each document, and selects the appropriate text field (either the original text or the translated text). The processed data is then saved to a new JSON fi