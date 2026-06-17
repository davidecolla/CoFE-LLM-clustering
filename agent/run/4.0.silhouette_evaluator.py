# © Copyright European Union - 2026

import json
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

def silhouette_evaluator(input_json_path, output_json_path, results_file_path, num_clusters=10):
    # Load the data from the JSON file
    with open(input_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Extract the translated text (or original if it's already in English)
    texts = [entry.get('text_en', entry['text']) for entry in data]
    
    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Calculate the silhouette scores for each sample
    silhouette_vals = silhouette_samples(X, clusters)
    
    # Calculate the average silhouette score
    avg_silhouette = silhouette_score(X, clusters)
    
    # Prepare the output data with cluster assignments
    output_data = [{"id": entry["id"], "cluster": str(cluster)} for entry, cluster in zip(data, clusters)]
    
    # Save the output data to a JSON file
    with open(output_json_path, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=4)
    
    # Save the silhouette scores and average silhouette score to a text file
    with open(results_file_path, 'w', encoding='utf-8') as file:
        file.write(f"Silhouette scores for each sample:\n{silhouette_vals}\n\n")
        file.write(f"Average silhouette score: {avg_silhouette}\n")
    
    # Print completion message
    print(f"Cluster evaluation completed. Results saved to {results_file_path} and {output_json_path}")

# Example usage
# silhouette_evaluator('../data/proposals_tiny_translation.json', 'output.json', 'results.txt')
```

This function performs the following tasks:
1. Loads the JSON data from the specified input file.
2. Extracts the translated text for each entry (or uses the original text if it's already in English).
3. Vectorizes the text data using TF-IDF.
4. Performs KMeans clustering with the specified number of clusters (10 by default).
5. Calculates the silhouette scores for each sample and the average silhouette score.
6. Saves the cluster assignments for each document to a JSON file.
7. Saves the silhouette scores and average silhouette score to a text file.
8. Prints a completion message indicating where the results are sav