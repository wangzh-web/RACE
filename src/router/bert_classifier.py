"""
Router: Paper B (QoSBERT) MC Dropout """
import torch
from typing import Tuple, List
import torch.nn.functional as F

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class UncertaintyRouter:
    def __init__(self, model_name: str = "distilbert-base-uncased", device: str = None):
        if not HAS_TRANSFORMERS:
            raise ImportError("Please install transformers: pip install transformers")
        
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(device)
        
    def predict_with_uncertainty(self, text: str, n_samples: int = 10) -> Tuple[float, float]:
        """
        MC Dropout         
        Returns:
            (pred_prob, uncertainty):         """
        self.model.train()
        
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.model(**inputs)
                prob = F.softmax(outputs.logits, dim=-1)[:, 1]
                probs.append(prob.item())
        
        mean_prob = sum(probs) / len(probs)
        variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
        
        return mean_prob, variance
    
    def train_on_data(self, train_data: List[Tuple[str, int]], epochs: int = 3):
        """
        Router
        
        train_data: [(question, is_correct), ...]
        """
        from torch.utils.data import DataLoader, Dataset
        from torch.optim import AdamW
        
        class QADataset(Dataset):
            def __init__(self, data, tokenizer):
                self.data = data
                self.tokenizer = tokenizer
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                text, label = self.data[idx]
                encoding = self.tokenizer(text, truncation=True, 
                                         max_length=512, padding="max_length")
                return {
                    "input_ids": torch.tensor(encoding["input_ids"]),
                    "attention_mask": torch.tensor(encoding["attention_mask"]),
                    "labels": torch.tensor(label)
                }
        
        dataset = QADataset(train_data, self.tokenizer)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)


if __name__ == "__main__":
    if HAS_TRANSFORMERS:
        router = UncertaintyRouter()
        prob, uncertainty = router.predict_with_uncertainty("What is 2+2?")
        print(f"Probability: {prob:.4f}")
        print(f"Uncertainty: {uncertainty:.4f}")
    else:
        print("transformers not installed")
