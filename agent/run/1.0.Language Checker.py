# © Copyright European Union - 2026

import json

def preprocess_data():
    # Load the dataset from the provided JSON file
    input_file_path = "../data/proposals_tiny.json"
    translation_file_path = "../data/proposals_tiny_translation.json"
    
    # Load the original dataset
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Load the translated dataset
    with open(translation_file_path, 'r', encoding='utf-8') as file:
        translated_data = json.load(file)
    
    # Create a dictionary for quick lookup of translated texts
    translated_dict = {item['id']: item for item in translated_data}
    
    # Preprocess the data by checking the language and using the translated text if necessary
    preprocessed_data = []
    for item in data:
        doc_id = item['id']
        language = item['language']
        
        if language == 'en':
            # If the document is already in English, use the original text
            text = item['text']
        else:
            # Otherwise, use the translated text
            text = translated_dict[doc_id]['text_en']
        
        # Append the preprocessed data to the list
        preprocessed_data.append({
            'id': doc_id,
            'text': text,
            'language': language
        })
    
    # Save the preprocessed data to a new JSON file
    output_file_path = "preprocessed_data.json"
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(preprocessed_data, file, ensure_ascii=False, indent=4)
    
    print(f"Preprocessed data saved to {output_file_path}")

# Call the function to preprocess the data
preprocess_data()
