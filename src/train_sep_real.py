#!/usr/bin/env python3
"""
train_sep_real.py - (SEP)
SEP1. TruthfulQA 2. Ground Truth
3. NLI 4. SEP MLP 
    python src/train_sep_real.py --num_samples 500 --num_generations 5

(8GB Mac):
    - 500 × 5 = ~1.5     - 2000 × 8 = ~8 ()
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================
# ============================================
DEFAULT_MODEL_PATH = "models/edge/Phi-3-mini-4k-instruct-q4.gguf"
DEFAULT_OUTPUT_DIR = "models/sep"
DEFAULT_DATA_DIR = "data/sep_training"


# ============================================
# ============================================
def download_truthfulqa(output_path: str = "data/truthfulqa.json") -> List[Dict]:
    from datasets import load_dataset
    
    print("TruthfulQA ...")
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    
    data = []
    for item in dataset:
        data.append({
            "question": item["question"],
            "best_answer": item["best_answer"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"],
        })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"{len(data)} {output_path}")
    return data


# ============================================
# ============================================
def generate_multiple_responses(
    model,
    questions: List[str],
    num_generations: int = 5,
    max_tokens: int = 128,
    temperature: float = 0.7
) -> List[List[str]]:
    all_responses = []
    
    for question in tqdm(questions, desc=""):
        responses = []
        for _ in range(num_generations):
            output = model(
                f"Question: {question}\nAnswer:",
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            responses.append(output['choices'][0]['text'].strip())
        all_responses.append(responses)
    
    return all_responses


# ============================================
# ============================================
def compute_semantic_entropy(
    responses: List[str],
    nli_model=None,
    nli_tokenizer=None
) -> float:
    """
        
        1. NLI     2.     3.     """
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    
    if len(responses) <= 1:
        return 0.0
    
    if nli_model is None:
        nli_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    embeddings = nli_model.encode(responses)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,
        metric='cosine',
        linkage='average'
    )
    
    try:
        labels = clustering.fit_predict(embeddings)
    except:
        return np.log(len(responses))
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    probs = counts / len(responses)
    
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return entropy


def generate_ground_truth(
    model,
    questions: List[str],
    num_generations: int = 5,
    batch_size: int = 10
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Ground Truth     
            hidden_states:         semantic_entropies:     """
    from sentence_transformers import SentenceTransformer
    
    nli_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    hidden_states = []
    semantic_entropies = []
    
    for i, question in enumerate(tqdm(questions, desc="Ground Truth")):
        responses = []
        for _ in range(num_generations):
            output = model(
                f"Question: {question}\nAnswer:",
                max_tokens=128,
                temperature=0.7,
                top_p=0.9,
            )
            responses.append(output['choices'][0]['text'].strip())
        
        embed = model.embed(f"Question: {question}\nAnswer:")
        hidden_states.append(embed)
        
        se = compute_semantic_entropy(responses, nli_model)
        semantic_entropies.append(se)
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(questions)}, {np.mean(semantic_entropies):.3f}")
    
    return hidden_states, semantic_entropies


# ============================================
# ============================================
class SemanticEntropyProbe(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def train_sep(
    hidden_states: List[np.ndarray],
    semantic_entropies: List[float],
    output_dir: str,
    epochs: int = 100,
    lr: float = 1e-3,
    val_split: float = 0.2
):
    
    X = np.array(hidden_states)
    y = np.array(semantic_entropies)
    
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min + 1e-10)
    
    n_val = int(len(X) * val_split)
    indices = np.random.permutation(len(X))
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    
    X_train, y_train = X[train_idx], y_norm[train_idx]
    X_val, y_val = X[val_idx], y_norm[val_idx]
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    input_dim = X.shape[1]
    model = SemanticEntropyProbe(input_dim=input_dim, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'hidden_dim': 128,
                'y_min': y_min,
                'y_max': y_max,
            }, os.path.join(output_dir, 'best_sep.pt'))
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
    
    print(f"\ndone! {best_val_loss:.4f}")
    print(f"{output_dir}/best_sep.pt")
    
    return model


# ============================================
# ============================================
def main():
    parser = argparse.ArgumentParser(description="(SEP)")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='GGUF ')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='')
    parser.add_argument('--num_generations', type=int, default=5,
                        help='')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='')
    parser.add_argument('--epochs', type=int, default=100,
                        help='')
    parser.add_argument('--skip_download', action='store_true',
                        help='()')
    args = parser.parse_args()
    
    print("=" * 60)
    print("(SEP) ")
    print("=" * 60)
    
    data_path = "data/truthfulqa.json"
    if not args.skip_download or not os.path.exists(data_path):
        data = download_truthfulqa(data_path)
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    questions = [item['question'] for item in data[:args.num_samples]]
    print(f"\n{len(questions)} ")
    
    print("\nLoading LLM ...")
    from llama_cpp import Llama
    model = Llama(
        model_path=args.model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        embedding=True,
        verbose=False
    )
    
    print(f"\nGround Truth ({args.num_generations} )...")
    hidden_states, semantic_entropies = generate_ground_truth(
        model, questions, num_generations=args.num_generations
    )
    
    cache_dir = "data/sep_training"
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "hidden_states.npy"), np.array(hidden_states))
    np.save(os.path.join(cache_dir, "semantic_entropies.npy"), np.array(semantic_entropies))
    print(f"\n{cache_dir}/")
    
    print("\nSEP ...")
    train_sep(hidden_states, semantic_entropies, args.output_dir, epochs=args.epochs)
    
    print("\n" + "=" * 60)
    print("done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
