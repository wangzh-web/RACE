"""
Download TriviaQA Dataset
"""
from datasets import load_dataset
import json
import os

def download_triviaqa():
    print("Downloading TriviaQA dataset...")
    
    
    try:
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", trust_remote_code=True)
    except Exception:
        print("Retrying with 'rc' config...")
        dataset = load_dataset("mandarjoshi/trivia_qa", "rc", trust_remote_code=True)
    
    def format_sample(sample):
        return {
            'question': sample['question'],
            'answer': sample['answer']['value'],
            'aliases': sample['answer']['aliases'],
        }
    
    calibration = [format_sample(s) for s in list(dataset['validation'])[:1000]]
    test = [format_sample(s) for s in list(dataset['validation'])[1000:6000]]
    
    os.makedirs('data/triviaqa', exist_ok=True)
    
    with open('data/triviaqa/calibration.json', 'w') as f:
        json.dump(calibration, f, indent=2)
    with open('data/triviaqa/test.json', 'w') as f:
        json.dump(test, f, indent=2)
    
    print(f"{len(calibration)} samples")
    print(f"{len(test)} samples")
    print(f"{calibration[0]}")

if __name__ == "__main__":
    download_triviaqa()
