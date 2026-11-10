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

print("Loading ALL France targets from test_source2 and test_source3...")
t0 = time.time()
tgt_list = []
for p in ['dataset/test/test_source2.tsv', 'dataset/test/test_source3.tsv']:
    for chunk in pd.read_csv(p, sep='\t', chunksize=200000):
        c_hits = chunk[chunk['country'] == 'France'][['entity_id', 'business_name', 'business_address']]
        if len(c_hits) > 0:
            tgt_list.append(c_hits)

df_tgt = pd.concat(tgt_list, ignore_index=True).drop_duplicates('entity_id')
print(f"Loaded {len(df_tgt):,} France targets in {time.time()-t0:.2f}s")

t0 = time.time()
df_tgt['norm_name'] = df_tgt['business_name'].apply(lambda x: normalize_text(x, is_address=False))
df_tgt['norm_addr'] = df_tgt['business_address'].apply(lambda x: normalize_text(x, is_address=True))
print(f"Normalized targets in {time.time()-t0:.2f}s")

# Load 1,000 France queries
df_q = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', nrows=20000)
df_q = df_q[df_q['country'] == 'France'].head(1000)
df_q['norm_name'] = df_q['business_name'].apply(lambda x: normalize_text(x, is_address=False))
df_q['norm_addr'] = df_q['business_address'].apply(lambda x: normalize_text(x, is_address=True))

q_ids = df_q['entity_id'].tolist()
q_names = df_q['norm_name'].tolist()
q_addrs = df_q['norm_addr'].tolist()
target_ids = df_tgt['entity_id'].values

class FastInvertedBlocker:
    def __init__(self, top_k: int = 20, max_postings: int = 600, top_tokens_per_field: int = 3):
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

t0 = time.time()
blocker = FastInvertedBlocker(top_k=20, max_postings=600, top_tokens_per_field=3)
blocker.fit_targets(target_ids, df_tgt['norm_name'].tolist(), df_tgt['norm_addr'].tolist())
print(f"Fit FastInvertedBlocker in {time.time()-t0:.2f}s, index size: {len(blocker.index):,} keys")

t0 = time.time()
cands, scores = blocker.query_batch(q_ids, q_names, q_addrs)
dt = time.time() - t0
print(f"Queried {len(q_ids)} France queries against 1.43M targets in {dt:.3f}s -> {len(q_ids)/dt:.1f} queries/sec!")
cands_per_q = np.mean([len(v) for v in cands.values()])
print(f"Average candidates per query: {cands_per_q:.1f}")
