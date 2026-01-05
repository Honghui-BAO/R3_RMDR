import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src/latent'))

import torch
from transformers import AutoTokenizer
from rmdr_sequential import RMDRDataset

class Args:
    def __init__(self, data_path, dataset, category="Toys_and_Games"):
        self.data_path = data_path
        self.dataset = dataset
        self.category = category

def test_loading():
    base_model = "Qwen/Qwen2.5-1.5B-Instruct" # Use a placeholder or actual local path if available
    # For testing, we can use a small model or just the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    except:
        print("Model not found, using generic tokenizer for test")
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|Thought|>"]})
    
    args = Args(
        data_path="/Users/honghuibao/Desktop/Baselines/RMDR/dataset",
        dataset="m_IOATBC-1.0-5-5"
    )
    
    try:
        dataset = RMDRDataset(args, tokenizer, mode='train', max_len=512)
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print("Sample keys:", sample.keys())
        print("Input IDs length:", len(sample['input_ids']))
        print("Decoded Input:", tokenizer.decode(sample['input_ids']))
        print("Labels (first 50 tokens):", sample['labels'][:50])
        
        # Check if Thought token is in decoded input
        decoded_str = tokenizer.decode(sample['input_ids'])
        if "<|Thought|>" in decoded_str:
            print("SUCCESS: Found <|Thought|> token in decoded output.")
        else:
            print("FAILURE: <|Thought|> token NOT found in decoded output.")
            
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    test_loading()
