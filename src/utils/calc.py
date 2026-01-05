# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# import transformers
# import torch
import os
import fire
import math
import json
import pandas as pd
import numpy as np
    
from tqdm import tqdm
def gao(path, item_path):
    print(path)
    print(item_path)
    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC=0

    f = open(f"{item_path}.txt", 'r')
    items = f.readlines()
    item_names = [_.split('\t')[0].strip("\"").strip(" ").strip('\n').strip('\"') for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])
    # print(item_dict) 
    ALLNDCG = np.zeros(5) # 1 3 5 10 20
    ALLHR = np.zeros(5)

    result_dict = dict()
    topk_list = [1, 3, 5, 10, 20]
    for p in path:
        result_dict[p] = {
            "NDCG": [],
            "HR": [],
        }
        f = open(p, 'r')
        import json
        test_data = json.load(f)
        f.close()
        
        domain_metrics = {} # domain -> {"NDCG": np.zeros(5), "HR": np.zeros(5), "count": 0}
        
        text = [ [_.strip(" \n").strip("\"").strip(" ") for _ in sample["predict"]] for sample in test_data]
        
        for index, sample in tqdm(enumerate(text)):
                domain = test_data[index].get("domain", "all")
                if domain not in domain_metrics:
                    domain_metrics[domain] = {"NDCG": np.zeros(5), "HR": np.zeros(5), "count": 0}
                domain_metrics[domain]["count"] += 1

                if type(test_data[index]['output']) == list:
                    target_item = test_data[index]['output'][0].strip("\"").strip(" ")
                else:
                    target_item = test_data[index]['output'].strip(" \n\"")
                minID = 1000000
                for i in range(len(sample)):
                    if sample[i] not in item_dict:
                        CC += 1
                    if sample[i] == target_item:
                        minID = i
                for k_idx, topk in enumerate(topk_list):
                    if minID < topk:
                        ALLNDCG[k_idx] = ALLNDCG[k_idx] + (1 / math.log(minID + 2))
                        ALLHR[k_idx] = ALLHR[k_idx] + 1
                        domain_metrics[domain]["NDCG"][k_idx] += (1 / math.log(minID + 2))
                        domain_metrics[domain]["HR"][k_idx] += 1
        
        print("\nResults by Domain:")
        for domain, metrics in domain_metrics.items():
            count = metrics["count"]
            print(f"Domain: {domain} (Count: {count})")
            print(f"NDCG@1,3,5,10,20: {metrics['NDCG'] / count / (1.0 / math.log(2))}")
            print(f"HR@1,3,5,10,20:   {metrics['HR'] / count}")
            print("-" * 20)

        print("\nOverall Results:")
        print(f"Total NDCG@1,3,5,10,20: {ALLNDCG / len(text) / (1.0 / math.log(2))}")
        print(f"Total HR@1,3,5,10,20:   {ALLHR / len(text)}")
        print(f"Total OOV Items: {CC}")

if __name__=='__main__':
    fire.Fire(gao)
