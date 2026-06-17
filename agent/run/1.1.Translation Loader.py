# © Copyright European Union - 2026

import json

def load_translations():
    # Define the path to the translation file
    translation_file_path = "../data/proposals_tiny_translation.json"
    
    # Load the translated dataset
    try:
        with open(translation_file_path, 'r', encoding='utf-8') as file:
            translated_data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {translation_file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {translation_file_path} is not a valid JSON file.")
        return None
    
    # Create a dictionary for quick lookup of translated texts
    translated_dict = {item['id']: item for item in translated_data}
    
    print(f"Translations loaded successfully. Total entries: {len(translated_dict)}")
    
    return translated_dict

# Example usage
# translated_dict = load_translations()
```

This function `load_translations` is responsible for loading the English translations of non-English texts from the specified JSON file. It handles potential errors such as file not found or invalid JSON format and returns a dictionary for quick lookup of translated tex