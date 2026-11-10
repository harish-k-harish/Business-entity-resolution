import os
import sys
import time
import math
import re
import heapq
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, 'code/business_entity_resolution/src')
from normalize import normalize_text

STOPWORDS = {
    'inc', 'corp', 'llc', 'ltd', 'co', 'pvt', 'prop', 'and', 'the', 'of', 'in', 'for',
    'sarl', 'sas', 'sasu', 'eurl', 'sa', 'sci', 'road', 'rd', 'street', 'st', 'avenue', 'ave',
    'blvd', 'dr', 'suite', 'ste', 'near', 'opp', 'bldg', 'floor', 'flr', 'sec', 'dist',
    'de', 'la', 'le', 'des', 'du', 'et', 'en', 'a', 'au', 'aux', 'd', 'l'
}
NUM_RE = re.compile(r'\b\d+\b')

def extract_tokens(name: str, addr: str):
    name_toks = [t for t in name.split() if t not in STOPWORDS and len(t) >= 2]
    nums = NUM_RE.findall(addr)
    addr_toks = [t for t in addr.split() if t not in STOPWORDS and len(t) >= 3 and not t.isdigit()]
    return name_toks, nums, addr_toks

class FastInvertedBlocker:
    def __init__(self, top_k: int = 20, max_postings: int = 800, top_tokens_per_field: int = 4):
        self.top_k = top_k
        self.max_postings = max_postings
        self.top_tokens_per_field = top_tokens_per_field
        self.target_ids = None
        self.index = {}
        self.idf_weights = {}

    def fit_targets(self, target_ids, norm_names, norm_addrs):
        self.target_ids = target_ids
        N_tgt = len(target_ids)
        raw_index = defaultdict(list)
        for idx, (name, addr) in enumerate(zip(norm_names, norm_addrs)):
            n_toks, nums, a_toks = extract_tokens(name, addr)
            for t in set(n_toks): raw_index[('name', t)].append(idx)
            for n in set(nums): raw_index[('num', n)].append(idx)
            for a in set(a_toks): raw_index[('addr', a)].append(idx)

        self.idf_weights = {}
        self.index = {}
        for key, post in raw_index.items():
            df = len(post)
            if 1 <= df <= self.max_postings:
                self.idf_weights[key] = math.log((N_tgt - df + 0.5) / (df + 0.5) + 1.0)
                self.index[key] = post

    def query_batch(self, query_ids, query_norm_names, query_norm_addrs):
        cands_dict = {}
        scores_dict = {}
        k = self.top_k
        tt = self.top_tokens_per_field

        for qid, q_name, q_addr in zip(query_ids, query_norm_names, query_norm_addrs):
            n_toks, nums, a_toks = extract_tokens(q_name, q_addr)
            sc_map = defaultdict(float)

            # Sort query tokens by IDF (highest weight first) and take top_tokens_per_field
            n_sorted = sorted([t for t in n_toks if ('name', t) in self.index], key=lambda t: self.idf_weights[('name', t)], reverse=True)[:tt]
            num_sorted = sorted([n for n in nums if ('num', n) in self.index], key=lambda n: self.idf_weights[('num', n)], reverse=True)[:tt]
            a_sorted = sorted([a for a in a_toks if ('addr', a) in self.index], key=lambda a: self.idf_weights[('addr', a)], reverse=True)[:tt]

            for t in n_sorted:
                key = ('name', t)
                w = 3.0 * self.idf_weights[key]
                for idx in self.index[key]:
                    sc_map[idx] += w

            for n in num_sorted:
                key = ('num', n)
                w = 2.5 * self.idf_weights[key]
                for idx in self.index[key]:
                    sc_map[idx] += w

            for a in a_sorted:
                key = ('addr', a)
                w = 1.5 * self.idf_weights[key]
                for idx in self.index[key]:
                    sc_map[idx] += w

            if len(sc_map) == 0:
                cands_dict[qid] = []
                continue
            elif len(sc_map) <= k:
                top_items = list(sc_map.items())
            else:
                top_items = heapq.nlargest(k, sc_map.items(), key=lambda x: x[1])

            top_indices = [x[0] for x in top_items]
            selected_ids = self.target_ids[top_indices].tolist()
            cands_dict[qid] = selected_ids
            for item, tid in zip(top_items, selected_ids):
                scores_dict[(qid, tid)] = min(1.0, item[1] / 30.0)

        return cands_dict, scores_dict

print("Loading validation set...")
# Load validation split from train data
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt_df = gt_df[gt_df['matched_entity_ids'].notna()]
gt_map = dict(zip(gt_df['source1_entity_id'], [set(str(x).split(',')) for x in gt_df['matched_entity_ids']]))

# Take 5,000 S1 queries from train_source1
s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=10000)
# Split by country: US and India
for c in ['India', 'US']:
    sub_s1 = s1_df[s1_df['country'] == c].head(2500).copy()
    sub_gt_keys = set(sub_s1['entity_id']) & set(gt_map.keys())
    total_true = sum(len(gt_map[k]) for k in sub_gt_keys)
    print(f"\n--- Testing Country {c}: {len(sub_s1)} queries, {len(sub_gt_keys)} with matches, {total_true} true pairs ---")
    
    tgt_list = []
    for p in ['dataset/train/train_source2.tsv', 'dataset/train/train_source3.tsv']:
        for chunk in pd.read_csv(p, sep='\t', chunksize=100000):
            c_hits = chunk[chunk['country'] == c][['entity_id', 'business_name', 'business_address']]
            if len(c_hits) > 0:
                tgt_list.append(c_hits)
            if sum(len(x) for x in tgt_list) > 300000:
                break
    df_tgt = pd.concat(tgt_list, ignore_index=True).drop_duplicates('entity_id')
    print(f"Loaded {len(df_tgt):,} targets for {c}")
    
    df_tgt['norm_name'] = df_tgt['business_name'].apply(lambda x: normalize_text(x, is_address=False))
    df_tgt['norm_addr'] = df_tgt['business_address'].apply(lambda x: normalize_text(x, is_address=True))
    sub_s1['norm_name'] = sub_s1['business_name'].apply(lambda x: normalize_text(x, is_address=False))
    sub_s1['norm_addr'] = sub_s1['business_address'].apply(lambda x: normalize_text(x, is_address=True))
    
    for max_p in [600, 1000, 1500]:
        for tt in [3, 4, 5]:
            blocker = FastInvertedBlocker(top_k=20, max_postings=max_p, top_tokens_per_field=tt)
            blocker.fit_targets(df_tgt['entity_id'].values, df_tgt['norm_name'].tolist(), df_tgt['norm_addr'].tolist())
            
            t0 = time.time()
            cands, _ = blocker.query_batch(sub_s1['entity_id'].tolist(), sub_s1['norm_name'].tolist(), sub_s1['norm_addr'].tolist())
            dt = time.time() - t0
            
            found = 0
            for qid in sub_gt_keys:
                found += len(set(cands.get(qid, [])) & gt_map[qid])
            recall = found / max(total_true, 1) * 100
            print(f"max_p={max_p:4d}, tt={tt}: Recall={recall:.2f}% ({found}/{total_true}), Speed={len(sub_s1)/dt:.1f} q/s ({dt:.2f}s)")
