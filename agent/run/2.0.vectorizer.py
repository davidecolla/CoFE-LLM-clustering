# © Copyright European Union - 2026

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def vectorize_text(input_file, output_file):
    """
    This function reads the JSON file containing the texts and their metadata,
    vectorizes the text data using TF-IDF, and saves the vectors to a new file.
    
    Parameters:
    - input_file: str, path to the input JSON file containing the texts and metadata.
    - output_file: str, path to the output file where the vectorized data will be saved.
    """
    # Load the data from the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract the text data for vectorization
    # If the text is not in English, use the translated text
    texts = []
    for entry in tqdm(data, desc="Extracting texts"):
        if entry['language'] == 'en':
            texts.append(entry['text'])
        else:
            texts.append(entry['text_en'])
    
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)  # Limiting to 1000 features for efficiency
    
    # Fit and transform the text data
    vectors = vectorizer.fit_transform(texts)
    
    # Save the vectorized data to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(vectors.toarray().tolist(), file)
    
    print(f"Vectorized data saved to {output_file}")

# Example usage:
# vectorize_text('../data/proposals_tiny.json', 'vectors.json')
```

This function reads the JSON file containing the texts and their metadata, vectorizes the text data using TF-IDF, and saves the vectors to a new file. The function uses the `TfidfVectorizer` from `sklearn.feature_extraction.text` to convert the text data into numerical vectors. The `tqdm` library is used to provide progress monitoring during the text extraction process. The vectorized data is saved as a JSON file containing a list of vecto