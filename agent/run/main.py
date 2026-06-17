# © Copyright European Union - 2026

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm

def json_loader(file_path):
    """Load JSON data from a specified file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
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

def translation_loader(file_path):
    """Load translation JSON data from a specified file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
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

def clusterer(input_file, translation_file, output_file, results_file):
    """Cluster the data into 10 clusters and evaluate using silhouette score."""
    # Load the dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Load the translated dataset
    with open(translation_file, 'r', encoding='utf-8') as f:
        translated_data = json.load(f)
    
    # Create a dictionary for quick access to translated texts
    translation_dict = {item['id']: item.get('text_en', item['text']) for item in translated_data}
    
    # Extract texts and ids
    texts = [translation_dict[item['id']] for item in data]
    ids = [item['id'] for item in data]
    
    # Vectorize the texts using TF-IDF
    print("Vectorizing texts using TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(texts)
    
    # Perform KMeans clustering
    print("Performing KMeans clustering...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Assign clusters to texts
    clusters = kmeans.labels_
    
    # Prepare the output data
    output_data = [{"id": id, "cluster": f"Cluster_{cluster}"} for id, cluster in zip(ids, clusters)]
    
    # Save the output data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    # Evaluate the clusters using silhouette score
    print("Evaluating clusters using silhouette score...")
    silhouette_avg = silhouette_score(X, clusters)
    sample_silhouette_values = silhouette_samples(X, clusters)
    
    # Prepare the results for saving
    results = {
        "average_silhouette_score": silhouette_avg,
        "silhouette_scores": {id: score for id, score in zip(ids, sample_silhouette_values)}
    }
    
    # Save the results to a text file
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Average Silhouette Score: {silhouette_avg}\n")
        for id, score in tqdm(zip(ids, sample_silhouette_values), desc="Writing silhouette scores"):
            f.write(f"ID: {id}, Silhouette Score: {score}\n")
    
    print(f"Clustering results saved to {output_file}")
    print(f"Evaluation results saved to {results_file}")

# Define file paths
input_file = '../data/proposals_tiny.json'
translation_file = '../data/proposals_tiny_translation.json'
output_file = 'output_clusters.json'
results_file = 'results.txt'

# Perform clustering and evaluation
clusterer(input_file, translation_file, output_file, results_file)
