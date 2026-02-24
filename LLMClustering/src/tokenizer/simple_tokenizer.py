from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class SimpleTokenizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=100,
            ngram_range=(1, 2)
        )
        
    def encode(self, texts, batch_size=32):
        """
        Encode texts into embeddings using TF-IDF vectorization.
        
        Args:
            texts (List[str]): List of texts to encode
            batch_size (int): Not used in this implementation
            
        Returns:
            np.ndarray: Array of embeddings
        """
        embeddings = self.vectorizer.fit_transform(texts)
        return embeddings.toarray()