"""
Download MMLU Dataset
"""
from datasets import load_dataset
import os

def download_mmlu():
    print("Downloading MMLU dataset...")
    
    dataset = load_dataset("cais/mmlu", "all")
    
    os.makedirs("data/mmlu", exist_ok=True)
    
    dataset.save_to_disk("data/mmlu")
    
    print(f"Available splits: {list(dataset.keys())}")
    if 'train' in dataset:
        print(f"Train: {len(dataset['train'])} samples")
    if 'validation' in dataset:
        print(f"Validation: {len(dataset['validation'])} samples")
    if 'test' in dataset:
        print(f"Test: {len(dataset['test'])} samples")
    if 'auxiliary_train' in dataset:
         print(f"Auxiliary Train: {len(dataset['auxiliary_train'])} samples")
    
    subjects = set(dataset['test']['subject'])
    print(f"Subjects ({len(subjects)}): {sorted(subjects)[:10]}...")
    
    return dataset


if __name__ == "__main__":
    download_mmlu()
