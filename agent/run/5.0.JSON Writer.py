# © Copyright European Union - 2026

import json
import os

def save_cluster_assignments(cluster_assignments, output_file='cluster_assignments.json'):
    """
    Save the cluster assignments to a JSON file.

    Parameters:
    - cluster_assignments: List of dictionaries, each containing 'id' and 'cluster' keys.
    - output_file: Name of the file to save the JSON data. Default is 'cluster_assignments.json'.
    
    Returns:
    - None
    """
    try:
        # Write the cluster assignments to a JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_assignments, f, ensure_ascii=False, indent=4)
        print(f"Cluster assignments saved successfully to {output_file}")
    except Exception as e:
        print(f"An error occurred while saving cluster assignments: {e}")

# Example usage:
# cluster_assignments = [
#     {"id": "doc1", "cluster": "Cluster1"},
#     {"id": "doc2", "cluster": "Cluster2"},
#     # ... more documents
# ]
# save_cluster_assignments(cluster_assignments)
```

This function takes a list of dictionaries containing cluster assignments and saves them to a JSON file. It includes error handling to provide feedback if something goes wrong during the file writing proce