# © Copyright European Union - 2026

import json

def translation_loader(file_path):
    """
    Load translation JSON data from a specified file path.

    Parameters:
    file_path (str): The path to the JSON file containing translations.

    Returns:
    list: A list of dictionaries containing the translation data.
    """
    try:
        # Open the file in read mode
        with open(file_path, 'r', encoding='utf-8') as file:
            # Load the JSON data from the file
            data = json.load(file)
        print(f"Translation data successfully loaded from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {file_path} is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Example usage:
# translation_data = translation_loader('../data/proposals_tiny_translation.json')
