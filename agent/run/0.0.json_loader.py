# © Copyright European Union - 2026

import json

def json_loader(file_path):
    """
    Load JSON data from a specified file path.

    Parameters:
    file_path (str): The path to the JSON file to be loaded.

    Returns:
    list: A list of dictionaries containing the JSON data.
    """
    try:
        # Open the file in read mode
        with open(file_path, 'r', encoding='utf-8') as file:
            # Load the JSON data from the file
            data = json.load(file)
        print(f"Data successfully loaded from {file_path}")
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
# data = json_loader('../data/proposals_tiny.json')
```

This function `json_loader` takes a file path as input, attempts to open and read the JSON file, and returns the data as a list of dictionaries. It includes error handling for common issues such as file not found and invalid JSON form