# © Copyright European Union - 2026

import json
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from tqdm import tqdm

def evaluate_clusters():
    # Load the dataset
    try:
        with open('../data/proposals_tiny_translation.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: The file proposals_tiny_translation.json was not found.")
        return
    except json.JSONDecodeError:
        print("Error: The file proposals_tiny_translation.json is not a valid JSON file.")
        return

    # Extract the translated text for clustering
    texts = [entry.get('text_en', entry['text']) for entry in data]

    # Vectorize the text data
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(texts)

    # Perform KMeans clustering with 10 clusters
    try:
        kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto')
        kmeans.fit(X)
    except Exception as e:
        print(f"Error during clustering: {e}")
        return

    # Predict the clusters
    clusters = kmeans.predict(X)

    # Calculate silhouette scores
    try:
        silhouette_avg = silhouette_score(X, clusters)
        sample_silhouette_values = silhouette_samples(X, clusters)
    except Exception as e:
        print(f"Error calculating silhouette scores: {e}")
        return

    # Prepare the output data with cluster information
    output_data = []
    for entry, cluster in zip(data, clusters):
        output_data.append({
            "id": entry["id"],
            "cluster": f"Cluster_{cluster}"
        })

    # Write the output data to a JSON file
    try:
        with open('clustered_proposals.json', 'w', encoding='utf-8') as output_file:
            json.dump(output_data, output_file, indent=4)
        print("Clustered proposals saved to clustered_proposals.json")
    except Exception as e:
        print(f"Error writing clustered_proposals.json: {e}")
        return

    # Write the silhouette scores to results.txt
    try:
        with open('results.txt', 'w', encoding='utf-8') as results_file:
            results_file.write(f"Average Silhouette Score: {silhouette_avg}\n")
            results_file.write("Silhouette Scores for Each Sample:\n")
            for score in tqdm(sample_silhouette_values, desc="Writing silhouette scores"):
                results_file.write(f"{score}\n")
        print("Silhouette scores saved to results.txt")
    except Exception as e:
        print(f"Error writing results.txt: {e}")
        return

# Call the function to execute the clustering and evaluation
evaluate_clusters()
