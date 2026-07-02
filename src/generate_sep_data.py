"""
(SEP)- GGUF(8GB )
"""
import json
import torch
import numpy as np
from tqdm import tqdm
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

MODEL_PATH = "models/edge/Phi-3-mini-4k-instruct-q4.gguf"
NUM_SAMPLES = 100
K = 5
TEMPERATURE = 0.7
SIMILARITY_THRESHOLD = 0.85
# ==============================

def load_models():
    print(f"Loading Phi-3 GGUF ({MODEL_PATH})...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1, 
        n_ctx=4096,
        embedding=True,
        verbose=False
    )
    
    print("Loading NLI ...")
    nli_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    return llm, nli_model

def get_hidden_state(llm, text):
    """
    (GGUF embedding)
    Llama.create_embedding token    """
    tokens = llm.tokenize(text.encode('utf-8'))
    if len(tokens) > 2048:
        tokens = tokens[-2048:]
        text = llm.detokenize(tokens).decode('utf-8', errors='ignore')
        
    embeddings = llm.create_embedding(text)
    
    # embeddings['data'][0]['embedding'] might be flattened list of all tokens
    # Reshape to (-1, 3072) and take last token
    emb_tensor = torch.tensor(embeddings['data'][0]['embedding'])
    if emb_tensor.numel() > 3072:
        emb_tensor = emb_tensor.view(-1, 3072)[-1, :]
    return emb_tensor.view(1, -1)

def generate_responses(llm, prompt, num_samples=K):
    responses = []
    for _ in range(num_samples):
        output = llm.create_completion(
            prompt,
            max_tokens=50,
            temperature=TEMPERATURE,
            stop=["\n\n", "Question:", "User:"],
            echo=False
        )
        response = output['choices'][0]['text'].strip()
        responses.append(response)
    
    return responses

def compute_semantic_entropy(responses, nli_model):
    if len(responses) <= 1:
        return 0.0
    
    valid_responses = [r for r in responses if r.strip()]
    if len(valid_responses) <= 1:
        return 0.0
    
    embeddings = nli_model.encode(valid_responses)
    sim_matrix = cosine_similarity(embeddings)
    
    n = len(valid_responses)
    visited = [False] * n
    clusters = []
    
    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and sim_matrix[i][j] >= SIMILARITY_THRESHOLD:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)
    
    K_total = len(valid_responses)
    cluster_probs = [len(c) / K_total for c in clusters]
    se = -sum(p * np.log(p + 1e-10) for p in cluster_probs)
    
    max_se = np.log(K_total)
    return se / max_se if max_se > 0 else 0

def main():
    print(f"=== SEP(GGUF) ===")
    
    if not os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH}")
        print("Running data/download_model_gguf.py")
        return

    llm, nli_model = load_models()
    
    with open('data/mmlu/calibration.json', 'r') as f:
        calibration_data = json.load(f)[:NUM_SAMPLES]
    
    print(f"\n{len(calibration_data)} samples")
    
    all_hidden_states = []
    all_semantic_entropies = []
    
    print("\n...")
    for idx, sample in enumerate(tqdm(calibration_data)):
        try:
            prompt = f"<|user|>\nQuestion: {sample['question']}\nChoices:\nA) {sample['choices'][0]}\nB) {sample['choices'][1]}\nC) {sample['choices'][2]}\nD) {sample['choices'][3]}\nAnswer with only the letter.<|end|>\n<|assistant|>\nThe answer is"
            
            hidden = get_hidden_state(llm, prompt)
            
            responses = generate_responses(llm, prompt, num_samples=K)
            
            se = compute_semantic_entropy(responses, nli_model)
            
            all_hidden_states.append(hidden)
            all_semantic_entropies.append(se)
            
            if (idx + 1) % 10 == 0:
                avg_se = np.mean(all_semantic_entropies)
                print(f"  [{idx+1}/{NUM_SAMPLES}] Mean SE: {avg_se:.3f}")
                
        except Exception as e:
            print(f"  Sample {idx} error: {e}")
            continue
    
    os.makedirs('data/sep', exist_ok=True)
    
    if len(all_hidden_states) > 0:
        hidden_tensor = torch.cat(all_hidden_states, dim=0)
        se_tensor = torch.tensor(all_semantic_entropies)
        
        torch.save({
            'hidden_states': hidden_tensor,
            'semantic_entropies': se_tensor,
        }, 'data/sep/training_data.pt')
        
        print(f"\n=== done ===")
        print(f"Save to: data/sep/training_data.pt")
        print(f"Count: {len(all_semantic_entropies)}")
        print(f"Hidden Dim: {hidden_tensor.shape}") # Should be 3072 for Phi-3
        print(f"SE Dist: mean={se_tensor.mean():.3f}, std={se_tensor.std():.3f}")
    else:
        print("Error: No samples processed")

if __name__ == "__main__":
    main()
