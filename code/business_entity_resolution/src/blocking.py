import math
import re
import heapq
from collections import defaultdict
import numpy as np

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

class CountryInvertedBlocker:
    """High-performance inverted index blocker with BM25/IDF field weighting and sublinear top-K retrieval."""
    def __init__(self, top_k: int = 20, max_postings: int = 1000, top_tokens_per_field: int = 4, **kwargs):
        self.top_k = top_k
        self.max_postings = max_postings
        self.top_tokens_per_field = top_tokens_per_field
        self.target_ids = None
        self.index = {}
        self.idf_weights = {}

    def fit_targets(self, target_ids: np.ndarray, target_norm_names: list, target_norm_addrs: list):
        """Build term-to-target inverted index with multi-field keys."""
        self.target_ids = target_ids
        N_tgt = len(target_ids)
        raw_index = defaultdict(list)

        for idx, (name, addr) in enumerate(zip(target_norm_names, target_norm_addrs)):
            n_toks, nums, a_toks = extract_tokens(name, addr)
            seen = set()
            for t in n_toks:
                if t not in seen:
                    raw_index[('name', t)].append(idx)
                    seen.add(t)
            for n in nums:
                if n not in seen:
                    raw_index[('num', n)].append(idx)
                    seen.add(n)
            for a in a_toks:
                if a not in seen:
                    raw_index[('addr', a)].append(idx)
                    seen.add(a)

        self.idf_weights = {}
        self.index = {}

        for key, post in raw_index.items():
            df = len(post)
            if 1 <= df <= self.max_postings:
                # BM25-style IDF weighting
                self.idf_weights[key] = math.log((N_tgt - df + 0.5) / (df + 0.5) + 1.0)
                self.index[key] = post

    def query_batch(self, query_ids: list, query_norm_names: list, query_norm_addrs: list):
        """Query batch of S1 records against inverted index and return top-k candidates."""
        cands_dict = {}
        scores_dict = {}
        k = self.top_k
        tt = self.top_tokens_per_field

        for qid, q_name, q_addr in zip(query_ids, query_norm_names, query_norm_addrs):
            n_toks, nums, a_toks = extract_tokens(q_name, q_addr)
            sc_map = defaultdict(float)

            # Sort query tokens by IDF (rarest, most discriminative first)
            n_sorted = sorted([t for t in n_toks if ('name', t) in self.index], key=lambda t: self.idf_weights[('name', t)], reverse=True)[:tt]
            num_sorted = sorted([n for n in nums if ('num', n) in self.index], key=lambda n: self.idf_weights[('num', n)], reverse=True)[:tt]
            a_sorted = sorted([a for a in a_toks if ('addr', a) in self.index], key=lambda a: self.idf_weights[('addr', a)], reverse=True)[:tt]

            # Name tokens (weight 3.0)
            for t in n_sorted:
                key = ('name', t)
                w = 3.0 * self.idf_weights[key]
                for idx in self.index[key]:
                    sc_map[idx] += w

            # Number tokens (weight 2.5)
            for n in num_sorted:
                key = ('num', n)
                w = 2.5 * self.idf_weights[key]
                for idx in self.index[key]:
                    sc_map[idx] += w

            # Address words (weight 1.5)
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

# Backward compatibility alias
CountrySparseBlocker = CountryInvertedBlocker
