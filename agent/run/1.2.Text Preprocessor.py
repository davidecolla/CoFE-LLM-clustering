# © Copyright European Union - 2026

import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Ensure the necessary NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Convert tokens to lower case
    tokens = [word.lower() for word in tokens]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Stem the words
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # Join the processed tokens back into a single string
    processed_text = ' '.join(stemmed_tokens)
    
    return processed_text

def preprocess_data():
    # Define the path to the preprocessed data file
    input_file_path = "preprocessed_data.json"
    output_file_path = "final_preprocessed_data.json"
    
    # Load the preprocessed data
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {input_file_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file {input_file_path} is not a valid JSON file.")
        return
    
    # Preprocess the text data
    print("Preprocessing text data...")
    for item in tqdm(data):
        item['text'] = preprocess_text(item['text'])
    
    # Save the final preprocessed data to a new JSON file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
    print(f"Final preprocessed data saved to {output_file_path}")

# Call the function to preprocess the data
preprocess_data()
