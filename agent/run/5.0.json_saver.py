# © Copyright European Union - 2026

import json

def json_saver(cluster_results, output_file='cluster_results.json'):
    """
    Save the cluster results to a JSON file.

    Parameters:
    - cluster_results (list): A list of dictionaries, each containing the document id and its cluster.
    - output_file (str): The name of the file where the results will be saved.

    Returns:
    - None
    """
    try:
        # Open the file in write mode
        with open(output_file, 'w', encoding='utf-8') as f:
            # Dump the cluster results into the file as a JSON array
            json.dump(cluster_results, f, ensure_ascii=False, indent=4)
        print(f"Cluster results successfully saved to {output_file}")
    except Exception as e:
        # Print an error message if something goes wrong
        print(f"An error occurred while saving the cluster results: {e}")

# Example usage:
# cluster_results = [{"id": "doc1", "cluster": "Cluster1"}, {"id": "doc2", "cluster": "Cluster2"}]
# json_saver(cluster_results)
