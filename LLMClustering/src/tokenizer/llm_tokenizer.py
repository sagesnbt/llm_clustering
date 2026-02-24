from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class LLMTokenizer:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def encode(self, texts, batch_size=32):
        """
        Encode texts into embeddings using the LLM.
        
        Args:
            texts (List[str]): List of texts to encode
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Array of embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                  return_tensors="pt", max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embeddings.append(outputs.last_hidden_state.mean(dim=1))
                
        return torch.cat(embeddings).numpy()