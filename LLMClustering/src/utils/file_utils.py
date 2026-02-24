import numpy as np
from typing import List, Dict, Any
import json
import os

def load_text_files(directory: str) -> List[str]:
    """
    Load text files from a directory.
    
    Args:
        directory (str): Path to directory containing text files
        
    Returns:
        List[str]: List of text contents
    """
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                texts.append(f.read())
    return texts

def save_clusters(texts: List[str], labels: np.ndarray, output_file: str):
    """
    Save clustering results to a JSON file.
    
    Args:
        texts (List[str]): Original texts
        labels (np.ndarray): Cluster labels
        output_file (str): Path to output JSON file
    """
    clusters: Dict[int, List[str]] = {}
    for text, label in zip(texts, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(text)
    
    with open(output_file, 'w') as f:
        json.dump(clusters, f, indent=2)