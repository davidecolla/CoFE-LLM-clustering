# © Copyright European Union - 2026

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm

def clusterer(input_file, translation_file, output_file, results_file):
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

# Example usage
# clusterer('../data/proposals_tiny.json',
#           '../data/proposals_tiny_translation.json',
#           'output_clusters.json',
#           'results.txt')
```

This function performs the following steps:
1. Loads the dataset and translated dataset from the provided JSON files.
2. Extracts the translated texts and corresponding IDs.
3. Vectorizes the texts using TF-IDF.
4. Performs KMeans clustering to form 10 clusters.
5. Assigns clusters to each text and prepares the output data.
6. Saves the output data to a JSON file.
7. Evaluates the clusters using silhouette scores and saves the evaluation results to a text fi