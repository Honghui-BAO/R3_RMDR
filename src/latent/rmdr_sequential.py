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
    def __init__(self, args, tokenizer, mode='train', max_len=512, sample=-1):
        super().__init__()
        self.args = args
        self.tokenizer = Tokenizer(tokenizer)
        self.mode = mode
        self.max_len = max_len
        
        # RMDR specific paths
        self.data_path = args.data_path
        self.dataset_name = args.dataset
        self.full_data_path = os.path.join(self.data_path, self.dataset_name)
        
        # Load item titles (optimized: prefer pre-tokenized/extracted list)
        self.local_path = f'local_dataset/{self.dataset_name}'
        self.item_titles = self._load_item_titles()
        
        # Load data splits
        self.data = self._load_split_data()
        
        # Optional Sampling for quick testing
        if sample > 0 and len(self.data) > sample:
            random.seed(42)
            self.data = random.sample(self.data, sample)
            
        self.length = len(self.data)
        
        # Category for instruction
        self.category = args.category if hasattr(args, 'category') else "items"
        
        # R3 specific: Thought token
        self.thought_token = "<|Thought|>"
        
        # Pre-calculate inputs to avoid bottleneck during conversion to HF Dataset
        self.inputs = []
        self._get_inputs()

    def _get_inputs(self):
        print(f"Tokenizing {self.mode} data ({len(self.data)} samples)...")
        for idx in tqdm(range(len(self.data)), desc=f"Processing {self.mode}"):
            self.inputs.append(self.pre_process(idx))

    def pre_process(self, idx):
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

    def _load_item_titles(self):
        title_path = os.path.join(self.local_path, "item_title_list.pkl")
        if os.path.exists(title_path):
            print(f"Loading item titles from {title_path}")
            return pickle.load(open(title_path, 'rb'))
            
        print(f"Title list not found, extracting from meta_datas.pkl (Slow)...")
        iid2asin = pickle.load(open(os.path.join(self.full_data_path, "iid2asin.pkl"), 'rb'))
        item_count = max(list(iid2asin.keys())) + 1
        item_metas = pickle.load(open(os.path.join(self.full_data_path, "meta_datas.pkl"), 'rb'))
        
        item_title_list = ['None'] * item_count
        for iid, asin in iid2asin.items():
            title = item_metas[asin]['title'] if (
                    'title' in item_metas[asin].keys() and item_metas[asin]['title']) else 'None'
            item_title_list[iid] = title
            
        # Cache for next time
        os.makedirs(self.local_path, exist_ok=True)
        pickle.dump(item_title_list, open(title_path, 'wb'))
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
            print(f"Loading split data from {path}")
            return pickle.load(open(path, 'rb'))
        else:
            print(f"Split data not found at {path}. Generating from raw data...")
            return self._generate_splits(local_path)[self.mode]

    def _generate_splits(self, local_path):
        import time
        rank = int(os.environ.get("RANK", "0"))
        
        if rank == 0:
            os.makedirs(local_path, exist_ok=True)
            raw_review_path = os.path.join(self.full_data_path, "review_datas.pkl")
            if not os.path.exists(raw_review_path):
                raise FileNotFoundError(f"Raw review data not found at {raw_review_path}")
                
            review_datas = pickle.load(open(raw_review_path, 'rb'))
            train_data, valid_data, test_data = [], [], []

            # Logic from RMDR data_sequential.py
            for user in tqdm(review_datas.keys(), desc='Splitting RMDR Data'):
                if not review_datas[user]: continue
                
                seq_iid_list = [review_datas[user][0][0]]
                seq_iid_cate_list = [review_datas[user][0][2]]

                for i in range(1, len(review_datas[user])):
                    target_iid = review_datas[user][i][0]
                    target_iid_cate = review_datas[user][i][2]
                    target_time = review_datas[user][i][3]
                    
                    # RMDR specific timestamps
                    if target_time < 1628643414042:
                        train_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                    elif target_time >= 1658002729837:
                        test_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                    else:
                        valid_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])

                    seq_iid_list = (seq_iid_list + [target_iid])[-10:] # max_seq_length=10 in RMDR
                    seq_iid_cate_list = (seq_iid_cate_list + [target_iid_cate])[-10:]

            # Save for future use
            pickle.dump(train_data, open(os.path.join(local_path, 'train_data.pkl'), 'wb'))
            pickle.dump(valid_data, open(os.path.join(local_path, 'valid_data.pkl'), 'wb'))
            pickle.dump(test_data, open(os.path.join(local_path, 'test_data.pkl'), 'wb'))
            
            return {'train': train_data, 'valid': valid_data, 'test': test_data}
        else:
            print(f"Rank {rank} waiting for rank 0 to generate data...")
            while not os.path.exists(os.path.join(local_path, 'test_data.pkl')):
                time.sleep(5)
            return {
                'train': pickle.load(open(os.path.join(local_path, 'train_data.pkl'), 'rb')),
                'valid': pickle.load(open(os.path.join(local_path, 'valid_data.pkl'), 'rb')),
                'test': pickle.load(open(os.path.join(local_path, 'test_data.pkl'), 'rb'))
            }

    def __len__(self):
        return self.length

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            example = self.data[i]
            seq_iid_list = example[0]
            target_iid = example[1]
            target_domain = example[3]
            target_title = self.item_titles[target_iid]
            
            history = ", ".join([f'"{self.item_titles[iid]}"' for iid in seq_iid_list])
            input_str = f"The user has enjoyed the following {self.category} recently: {history}"

            temp.append({
                "input": input_str,
                "output": target_title,
                "domain": target_domain
            })
        return temp

    def _generate_prompt(self, seq_iid_list, target_title):
        history = ", ".join([f'"{self.item_titles[iid]}"' for iid in seq_iid_list])
        input_str = f"The user has enjoyed the following {self.category} recently: {history}"
        
        instruction = f"### Instruction:\nGiven a list of {self.category} the user recently enjoyed, please predict the next {self.category} the user might like.\n\n"
        
        user_input = f"### User Input:\n{input_str}\n\n### Response:\n"
        
        # R3's protocol: Instruction + User Input + Thought Token
        full_prompt = instruction + user_input + self.thought_token
        return full_prompt, target_title

    def __getitem__(self, idx):
        return self.inputs[idx]
