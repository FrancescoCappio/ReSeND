import random
import os
import numpy as np
from tqdm import tqdm

def preprocess_dataset(files):
    lines = [f.strip() for f in files]
    files = []
    lbls = []
    for l in lines:
        file_path, lbl = l.split()
        files.append(file_path)
        lbls.append(int(lbl))

    lbl_set = set(lbls)
    indices = np.arange(len(lbls))
    lbls = np.array(lbls)

    cat_indices = {}
    for lbl in lbl_set:
        cat_indices[lbl] = indices[lbls==lbl]
    
    return files, lbls, cat_indices, lbl_set


def create_pairs_source_v2(path_to_txt, enable_class_balancing=True, neg_to_pos_ratio=1):

    print("Generate source pairs")
    pairs = []

    # for each category we count occurrences
    cat_count = {}

    with open(path_to_txt) as input_file:
        file_names = input_file.readlines()

    files_paths, lbls, cat_indices, lbl_set = preprocess_dataset(file_names)

    max_cardinality = 0
    tuple_lbl_set = tuple(lbl_set)
    for k, v in cat_indices.items():
        cat_count[k] = len(v)
        if len(v) > max_cardinality:
            max_cardinality = len(v)

    for list_idx in tqdm(range(len(files_paths))):
        file_1, category = files_paths[list_idx], lbls[list_idx]

        # other file same category
        idx = np.random.choice(cat_indices[category])
        other_file = files_paths[idx]
        pairs.append(file_1+' '+other_file+' '+str(category)+' '+str(0)+'\n')

        # other file different category
        for neg in range(neg_to_pos_ratio):

            different_cat = category
            while different_cat == category:
                different_cat = random.choice(tuple_lbl_set)
            idx = np.random.choice(cat_indices[different_cat])

            other_file = files_paths[idx]
            pairs.append(file_1+' '+other_file+' '+str(category)+' '+str(1)+'\n')

    if enable_class_balancing:
        print("Generate additional pairs for class balancing")
        for lbl in tqdm(range(len(cat_count))):
            while cat_count[lbl] < max_cardinality:
                idx_1 = np.random.choice(cat_indices[lbl])
                file_1 = files_paths[idx_1]
                idx_2 = np.random.choice(cat_indices[lbl])
                file_2 = files_paths[idx_2]

                pairs.append(file_1+' '+file_2+' '+str(lbl)+' '+str(0)+'\n')

                for neg in range(neg_to_pos_ratio):
                    different_cat = lbl
                    while different_cat == lbl:
                        different_cat = random.choice(tuple_lbl_set)
                    idx_2 = np.random.choice(cat_indices[different_cat])

                    file_2 = files_paths[idx_2]
                    pairs.append(file_1+' '+file_2+' '+str(lbl)+' '+str(1)+'\n')

                cat_count[lbl] += 1
    return pairs


