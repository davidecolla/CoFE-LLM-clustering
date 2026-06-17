# © Copyright European Union - 2026

import json
import os

def load_data(file_path):
    """
    Load the dataset from a JSON file.

    Parameters:
    file_path (str): The path to the JSON file containing the dataset.

    Returns:
    list: A list of dictionaries, each representing a document with 'text', 'id', and 'language' keys.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Open and load the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        raise ValueError(f"The file {file_path} is not a valid JSON file.")
    
    # Return the loaded data
    return data

# Example usage:
# data = load_data('../data/proposals_tiny.json')
# print(data[:5])  # Print the first 5 entries to verify
