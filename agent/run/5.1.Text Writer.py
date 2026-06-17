# © Copyright European Union - 2026

def save_silhouette_scores(silhouette_scores, average_silhouette, output_file='results.txt'):
    """
    Save the silhouette scores and average silhouette score to a text file.

    Parameters:
    - silhouette_scores: List of silhouette scores for each sample.
    - average_silhouette: The average silhouette score for all samples.
    - output_file: Name of the file to save the silhouette scores. Default is 'results.txt'.
    
    Returns:
    - None
    """
    try:
        # Write the silhouette scores and average silhouette score to a text file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Silhouette scores for each sample:\n")
            for i, score in enumerate(silhouette_scores):
                f.write(f"Sample {i+1}: {score}\n")
            f.write("\nAverage silhouette score: {average_silhouette}\n")
        print(f"Silhouette scores saved successfully to {output_file}")
    except Exception as e:
        print(f"An error occurred while saving silhouette scores: {e}")

# Example usage:
# silhouette_scores = [0.5, 0.6, 0.7, 0.8, 0.9]
# average_silhouette = 0.7
# save_silhouette_scores(silhouette_scores, average_silhouette)
