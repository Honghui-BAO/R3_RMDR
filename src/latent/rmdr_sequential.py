import pickle
import torch
from torch.utils.data import Dataset
import os
import random
from tqdm import tqdm
from typing import List, Tuple

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        if not t:
            return t
        while t[0] == self.bos_id:
            t = t[1:]
            if not t: break
        if t:
            while t[-1] == self.eos_id:
                t = t[:-1]
                if not t: break

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

class RMDRDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train', max_len=512):
        super().__init__()
        self.args = args
        self.tokenizer = Tokenizer(tokenizer)
        self.mode = mode
        self.max_len = max_len
        
        # RMDR specific paths
        self.data_path = args.data_path
        self.dataset_name = args.dataset
        self.full_data_path = os.path.join(self.data_path, self.dataset_name)
        
        # Load item titles
        self.iid2asin = pickle.load(open(os.path.join(self.full_data_path, "iid2asin.pkl"), 'rb'))
        self.item_count = max(list(self.iid2asin.keys())) + 1
        
        self.item_metas = pickle.load(open(os.path.join(self.full_data_path, "meta_datas.pkl"), 'rb'))
        self.item_titles = self._load_item_titles()
        
        # Load data splits
        self.data = self._load_split_data()
        self.length = len(self.data)
        
        # Category for instruction
        self.category = args.category if hasattr(args, 'category') else "items"
        
        # R3 specific: Thought token
        # We assume tokenizer already has <|Thought|> added from latent_attention_train.py
        self.thought_token = "<|Thought|>"
        
    def _load_item_titles(self):
        item_title_list = ['None'] * self.item_count
        for iid, asin in self.iid2asin.items():
            title = self.item_metas[asin]['title'] if (
                    'title' in self.item_metas[asin].keys() and self.item_metas[asin]['title']) else 'None'
            item_title_list[iid] = title
        return item_title_list

    def _load_split_data(self):
        # Local dataset path from RMDR logic
        local_path = f'local_dataset/{self.dataset_name}'
        if self.mode == 'train':
            file_name = 'train_data.pkl'
        elif self.mode == 'valid':
            file_name = 'valid_data.pkl'
        else:
            file_name = 'test_data.pkl'
        
        path = os.path.join(local_path, file_name)
        if os.path.exists(path):
            return pickle.load(open(path, 'rb'))
        else:
            # If local split doesn't exist, we might need to implement the split logic from RMDR
            # or expect it to be pre-generated. For now, let's assume it exists or raise error.
            raise FileNotFoundError(f"RMDR split data not found at {path}. Please run RMDR preprocessing first.")

    def __len__(self):
        return self.length

    def _generate_prompt(self, seq_iid_list, target_title):
        history = ", ".join([f'"{self.item_titles[iid]}"' for iid in seq_iid_list])
        input_str = f"The user has enjoyed the following {self.category} recently: {history}"
        
        instruction = f"### Instruction:\nGiven a list of {self.category} the user recently enjoyed, please predict the next {self.category} the user might like.\n\n"
        
        user_input = f"### User Input:\n{input_str}\n\n### Response:\n"
        
        # R3's protocol: Instruction + User Input + Thought Token
        full_prompt = instruction + user_input + self.thought_token
        return full_prompt, target_title

    def __getitem__(self, idx):
        # RMDR data format: [seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate]
        example = self.data[idx]
        seq_iid_list = example[0]
        target_iid = example[1]
        target_title = self.item_titles[target_iid]
        
        prompt_str, target_str = self._generate_prompt(seq_iid_list, target_title)
        
        # Tokenize prompt (BOS=True, EOS=False)
        prompt_ids = self.tokenizer.encode(prompt_str, bos=True, eos=False)
            
        # Tokenize target (BOS=False, EOS=True)
        # We need a newline before target as per R3's LatentRDataset
        target_ids = self.tokenizer.encode('\n' + target_str, bos=False, eos=True)
        
        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids
        
        # Truncate/Pad
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
            attention_mask = attention_mask[-self.max_len:]
            labels = labels[-self.max_len:]
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
