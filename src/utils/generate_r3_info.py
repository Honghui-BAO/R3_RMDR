import pickle
import os
import fire

def main(data_path, dataset, output_file):
    full_path = os.path.join(data_path, dataset)
    iid2asin = pickle.load(open(os.path.join(full_path, "iid2asin.pkl"), 'rb'))
    item_count = max(list(iid2asin.keys())) + 1
    item_metas = pickle.load(open(os.path.join(full_path, "meta_datas.pkl"), 'rb'))
    
    with open(output_file, 'w') as f:
        for i in range(item_count):
            asin = iid2asin.get(i)
            if asin and asin in item_metas:
                title = item_metas[asin].get('title', 'None')
                if not title: title = 'None'
            else:
                title = 'None'
            # R3 format: title \t category \t other_info
            # We only really need the title in the first column
            f.write(f"{title}\tNone\tNone\n")
    print(f"Generated {output_file}")

if __name__ == "__main__":
    fire.Fire(main)
