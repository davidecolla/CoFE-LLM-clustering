# © Copyright European Union - 2026

import json

def silhouette_saver(silhouette_scores, average_silhouette, output_file='results.txt'):
    """
    Save the silhouette evaluation to a file named results.txt.

    Parameters:
    - silhouette_scores (list): A list of silhouette scores for each sample.
    - average_silhouette (float): The average silhouette score.
    - output_file (str): The name of the file where the results will be saved.

    Returns:
    - None
    """
    try:
        # Open the file in write mode
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write the silhouette scores for each sample
            f.write("Silhouette Scores for each sample:\n")
            for score in silhouette_scores:
                f.write(f"{score}\n")
            
            # Write the average silhouette score
            f.write(f"\nAverage Silhouette Score: {average_silhouette}\n")
        
        print(f"Silhouette evaluation successfully saved to {output_file}")
    except Exception as e:
        # Print an error message if something goes wrong
        print(f"An error occurred while saving the silhouette evaluation: {e}")

# Example usage:
# silhouette_scores = [0.1, 0.2, 0.3]
# average_silhouette = 0.2
# silhouette_saver(silhouette_scores, average_silhouette)
