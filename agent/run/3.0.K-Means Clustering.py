# © Copyright European Union - 2026

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm

def cluster_data():
    # Load the dataset
    with open('../data/proposals_tiny_translation.json', 'r') as f:
        data = json.load(f)
    
    # Extract the text data for clustering
    texts = [entry['text_en'] if 'text_en' in entry else entry['text'] for entry in data]
    
    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X = vectorizer.fit_transform(texts)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Prepare the output data with clusters
    output_data = [{"id": entry["id"], "cluster": f"Cluster_{cluster}"} for entry, cluster in zip(data, clusters)]
    
    # Save the output data to a JSON file
    with open('output_clusters.json', 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # Evaluate the clustering using silhouette score
    silhouette_avg = silhouette_score(X, clusters)
    sample_silhouette_values = silhouette_samples(X, clusters)
    
    # Prepare the results for evaluation
    results = f"Average Silhouette Score: {silhouette_avg}\n"
    results += "Silhouette Score for each sample:\n"
    for i, silhouette_value in enumerate(tqdm(sample_silhouette_values, desc="Calculating silhouette scores")):
        results += f"Sample {i}: {silhouette_value}\n"
    
    # Save the evaluation results to a text file
    with open('results.txt', 'w') as f:
        f.write(results)
    
    print("Clustering complete. Output saved to output_clusters.json and evaluation results to results.txt.")

# Run the function
cluster_data()
