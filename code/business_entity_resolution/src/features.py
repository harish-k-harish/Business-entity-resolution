import re
import numpy as np
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz

NUM_REGEX = re.compile(r'\b\d+\b')

FEATURE_NAMES = [
    'jw_name',
    'tsr_name',
    'tset_name',
    'ratio_name',
    'tsr_addr',
    'tset_addr',
    'jaccard_addr',
    'num_exact',
    'num_overlap',
    'name_len_diff',
    'comb_sparse_score'
]

def extract_pair_features(n1: str, a1: str, n2: str, a2: str, comb_sparse_score: float = 0.0) -> list:
    """Extract pairwise similarity features between query (n1, a1) and candidate (n2, a2)."""
    # Name features
    jw_name = JaroWinkler.similarity(n1, n2)
    tsr_name = fuzz.token_sort_ratio(n1, n2) / 100.0
    tset_name = fuzz.token_set_ratio(n1, n2) / 100.0
    ratio_name = fuzz.ratio(n1, n2) / 100.0
    name_len_diff = abs(len(n1) - len(n2)) / max(1, max(len(n1), len(n2)))
    
    # Address features
    tsr_addr = fuzz.token_sort_ratio(a1, a2) / 100.0
    tset_addr = fuzz.token_set_ratio(a1, a2) / 100.0
    
    # Token Jaccard on address
    s1_toks = set(a1.split())
    s2_toks = set(a2.split())
    union_len = len(s1_toks | s2_toks)
    jaccard_addr = (len(s1_toks & s2_toks) / union_len) if union_len > 0 else 0.0
    
    # Numeric sequence matching (street numbers, PIN/postal codes)
    nums1 = set(NUM_REGEX.findall(a1))
    nums2 = set(NUM_REGEX.findall(a2))
    if nums1 and nums2:
        num_exact = 1.0 if nums1 == nums2 else 0.0
        num_overlap = len(nums1 & nums2) / len(nums1 | nums2)
    elif not nums1 and not nums2:
        num_exact = 0.5
        num_overlap = 0.5
    else:
        num_exact = 0.0
        num_overlap = 0.0
        
    return [
        jw_name,
        tsr_name,
        tset_name,
        ratio_name,
        tsr_addr,
        tset_addr,
        jaccard_addr,
        num_exact,
        num_overlap,
        name_len_diff,
        comb_sparse_score
    ]
