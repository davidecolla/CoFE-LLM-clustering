# © Copyright European Union - 2026

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def vectorize_data(input_file_path, output_file_path):
    """
    This function reads the JSON file containing the documents,
    vectorizes the text data using TF-IDF, and saves the vectors to a file.
    
    Parameters:
    - input_file_path: Path to the input JSON file.
    - output_file_path: Path to save the vectorized data.
    """
    # Load the data from the JSON file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract the text data for vectorization
    texts = [entry.get('text_en', entry['text']) for entry in data]
    
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to 5000 features for efficiency
    
    # Vectorize the text data
    print("Vectorizing the text data using TF-IDF...")
    X = vectorizer.fit_transform(texts)
    
    # Convert the sparse matrix to a dense format for easier handling
    X_dense = X.toarray()
    
    # Save the vectorized data to a file
    np.save(output_file_path, X_dense)
    print(f"Vectorized data saved to {output_file_path}")

# Example usage:
# vectorize_data('../data/proposals_tiny.json', 'vectorized_data.npy')
```

This function reads the input JSON file containing the documents, vectorizes the text data using TF-IDF, and saves the resulting vectors to a NumPy file. The vectorization process uses a maximum of 5000 features for efficiency. The function prints progress messages to inform the user of the steps being tak