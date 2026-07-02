"""
Semantic Entropy: Farquhar et al., Nature 2024

- LLM - - 
SE(x) = -Σ p(c|x) log p(c|x)
"""
import numpy as np
from typing import List, Tuple
from collections import Counter

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SemanticEntropyRouter:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        embedding_model:         """
        if HAS_SENTENCE_TRANSFORMERS:
            self.embedder = SentenceTransformer(embedding_model)
        else:
            self.embedder = None
            print("Warning: sentence-transformers not installed. Using simple hash clustering.")
        self.similarity_threshold = 0.85
        
    def compute_semantic_entropy(self, responses: List[str]) -> float:
        """
                
        Args:
            responses: LLM         
        Returns:
            semantic_entropy: ()
        """
        if len(responses) == 0:
            return float('inf')
        if len(responses) == 1:
            return 0.0
        
        if self.embedder is not None:
            return self._compute_with_embeddings(responses)
        else:
            return self._compute_with_simple_clustering(responses)
    
    def _compute_with_embeddings(self, responses: List[str]) -> float:
        embeddings = self.embedder.encode(responses)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - self.similarity_threshold,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        
        unique, counts = np.unique(labels, return_counts=True)
        probs = counts / len(responses)
        
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return entropy
    
    def _compute_with_simple_clustering(self, responses: List[str]) -> float:
        normalized = [r.strip().lower()[:50] for r in responses]
        counter = Counter(normalized)
        probs = np.array(list(counter.values())) / len(responses)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy
    
    def predict_uncertainty(self, question: str, edge_model, n_samples: int = 5) -> Tuple[float, List[str]]:
        """
                
        Args:
            question:             edge_model: LLM (generate )
            n_samples:         
        Returns:
            (uncertainty, responses):         """
        responses = []
        for _ in range(n_samples):
            result = edge_model.generate(question)
            if isinstance(result, dict):
                responses.append(result.get("response", ""))
            else:
                responses.append(str(result))
        
        uncertainty = self.compute_semantic_entropy(responses)
        
        max_entropy = np.log(n_samples)
        normalized_uncertainty = min(uncertainty / max_entropy, 1.0) if max_entropy > 0 else 0.0
        
        return normalized_uncertainty, responses
    
    def get_majority_answer(self, responses: List[str]) -> str:
        if not responses:
            return ""
        return Counter(responses).most_common(1)[0][0]


if __name__ == "__main__":
    router = SemanticEntropyRouter()
    
    consistent = ["Paris", "Paris", "Paris", "Paris", "Paris"]
    print(f"Consistent responses entropy: {router.compute_semantic_entropy(consistent):.4f}")
    
    inconsistent = ["Paris", "London", "Berlin", "Tokyo", "Rome"]
    print(f"Inconsistent responses entropy: {router.compute_semantic_entropy(inconsistent):.4f}")
