#!/usr/bin/env python3
"""
train_sep_mmlu.py - MMLUSEP
SEP"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
from datasets import load_from_disk
import warnings
warnings.filterwarnings('ignore')


class SemanticEntropyProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"\n{'='*60}")
    print(f"Loading{model_name} (FP16)")
    print('='*60)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ LoadingdoneVRAM: {model.get_memory_footprint() / 1e9:.2f} GB\n")
    return model, tokenizer


def compute_semantic_entropy(responses: List[str]) -> float:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import AgglomerativeClustering
    
    if len(responses) <= 1:
        return 0.0
    
    responses = [r for r in responses if r.strip()]
    if len(responses) <= 1:
        return 0.0
    
    encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = encoder.encode(responses)
    
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(responses)
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        return entropy
    except Exception:
        return float(np.log(len(responses)))


def generate_response(model, tokenizer, question: str, choices: List[str], temperature: float = 0.7) -> str:
    prompt = f"""Answer the following multiple choice question with only the letter (A, B, C, or D).

Question: {question}
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def get_hidden_states(model, tokenizer, question: str, choices: List[str]) -> np.ndarray:
    prompt = f"Question: {question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden = outputs.hidden_states[-1][:, -1, :].cpu().float().numpy()
    return hidden.squeeze()


def train_sep_from_data(X: np.ndarray, y: np.ndarray, output_dir: str, epochs: int = 100):
    print("\n" + "="*60)
    print("SEP ")
    print("="*60)
    
    y_min, y_max = float(y.min()), float(y.max())
    print(f"[{y_min:.4f}, {y_max:.4f}]")
    
    if y_max > y_min:
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = y * 0 + 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_norm, dtype=torch.float32, device=device).unsqueeze(1)
    
    n_samples = len(X)
    n_val = max(1, int(n_samples * 0.2))
    indices = list(range(n_samples))
    np.random.shuffle(indices)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    
    X_train, y_train = X_t[train_idx], y_t[train_idx]
    X_val, y_val = X_t[val_idx], y_t[val_idx]
    
    print(f"samples: {len(train_idx)}, samples: {len(val_idx)}")
    
    input_dim = X.shape[1]
    sep = SemanticEntropyProbe(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(sep.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(epochs):
        sep.train()
        optimizer.zero_grad()
        pred = sep(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        
        sep.eval()
        with torch.no_grad():
            val_pred = sep(X_val)
            val_loss = criterion(val_pred, y_val)
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(sep.state_dict(), os.path.join(output_dir, 'best_sep.pt'))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train: {loss.item():.4f}, Val: {val_loss.item():.4f}")
    
    print(f"\n✓ done! {best_val_loss:.4f}")
    print(f"✓ {output_dir}/best_sep.pt")
    
    meta = {
        'input_dim': input_dim,
        'hidden_dim': 128,
        'y_min': y_min,
        'y_max': y_max,
        'num_samples': n_samples,
        'dataset': 'mmlu',
    }
    with open(os.path.join(output_dir, 'sep_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    return best_val_loss


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--num_generations', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default='models/sep_mmlu')
    parser.add_argument('--use_cache', action='store_true', help='')
    args = parser.parse_args()
    
    print("="*60)
    print("(SEP) - MMLU ")
    print("="*60)
    
    cache_file = os.path.join(args.output_dir, 'training_data.npz')
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.use_cache and os.path.exists(cache_file):
        print(f"✓ Loading{cache_file}")
        data = np.load(cache_file)
        X = data['hidden_states']
        y = data['entropies']
        print(f"  - samples{len(X)}, {X.shape[1]}")
    else:
        print("LoadingMMLU...")
        try:
            dataset = load_from_disk("data/mmlu/auxiliary_train")
            data_list = [{"question": item["question"], "choices": item["choices"]} 
                        for item in dataset]
            print(f"✓ ArrowLoading {len(data_list)} ")
        except:
            with open("data/mmlu/test.json", 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            data_list = all_data[-args.num_samples:]
            print(f"✓ test.jsonLoading {len(data_list)} ")
        
        np.random.seed(42)
        if len(data_list) > args.num_samples:
            indices = np.random.choice(len(data_list), args.num_samples, replace=False)
            data_list = [data_list[i] for i in indices]
        
        print(f"{len(data_list)} \n")
        
        model, tokenizer = load_model()
        
        print("(+ )...")
        hidden_states = []
        entropies = []
        
        for i, item in enumerate(tqdm(data_list, desc="")):
            question = item['question']
            choices = item['choices']
            
            hidden = get_hidden_states(model, tokenizer, question, choices)
            hidden_states.append(hidden)
            
            responses = []
            for _ in range(args.num_generations):
                resp = generate_response(model, tokenizer, question, choices, temperature=0.7)
                responses.append(resp)
            
            se = compute_semantic_entropy(responses)
            entropies.append(se)
            
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(data_list)}, {np.mean(entropies):.3f}")
        
        X = np.array(hidden_states, dtype=np.float32)
        y = np.array(entropies, dtype=np.float32)
        
        np.savez(cache_file, hidden_states=X, entropies=y)
        print(f"\n✓ {cache_file}")
        
        del model, tokenizer
        torch.cuda.empty_cache()
    
    train_sep_from_data(X, y, args.output_dir, args.epochs)


if __name__ == "__main__":
    main()
