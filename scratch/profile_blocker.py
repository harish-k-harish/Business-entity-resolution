import sys
import time
import cProfile
import pstats
import pandas as pd
import numpy as np

sys.path.insert(0, 'code/business_entity_resolution/src')
from normalize import normalize_text
from blocking import CountryInvertedBlocker
from features import extract_pair_features
from model import load_matcher_model

print("Loading 200,000 France targets...")
tgt_list = []
for p in ['dataset/test/test_source2.tsv', 'dataset/test/test_source3.tsv']:
    for chunk in pd.read_csv(p, sep='\t', chunksize=200000):
        c_hits = chunk[chunk['country'] == 'France'][['entity_id', 'business_name', 'business_address']]
        if len(c_hits) > 0:
            tgt_list.append(c_hits)
        if sum(len(x) for x in tgt_list) > 200000:
            break

df_tgt = pd.concat(tgt_list, ignore_index=True).drop_duplicates('entity_id').head(200000)
df_tgt['norm_name'] = df_tgt['business_name'].apply(lambda x: normalize_text(x, is_address=False))
df_tgt['norm_addr'] = df_tgt['business_address'].apply(lambda x: normalize_text(x, is_address=True))

blocker = CountryInvertedBlocker(top_k=20)
blocker.fit_targets(
    target_ids=df_tgt['entity_id'].values,
    target_norm_names=df_tgt['norm_name'].tolist(),
    target_norm_addrs=df_tgt['norm_addr'].tolist()
)

target_name_dict = dict(zip(df_tgt['entity_id'].values, df_tgt['norm_name']))
target_addr_dict = dict(zip(df_tgt['entity_id'].values, df_tgt['norm_addr']))

df_q = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', nrows=10000)
df_q = df_q[df_q['country'] == 'France'].head(500)
df_q['norm_name'] = df_q['business_name'].apply(lambda x: normalize_text(x, is_address=False))
df_q['norm_addr'] = df_q['business_address'].apply(lambda x: normalize_text(x, is_address=True))

s1_ids = df_q['entity_id'].tolist()
s1_names = df_q['norm_name'].tolist()
s1_addrs = df_q['norm_addr'].tolist()

def run_bench():
    t0 = time.time()
    cands_dict, scores_dict = blocker.query_batch(s1_ids, s1_names, s1_addrs)
    t_block = time.time() - t0
    
    t0 = time.time()
    batch_pairs = []
    batch_feats = []
    for qid, q_name, q_addr in zip(s1_ids, s1_names, s1_addrs):
        c_list = cands_dict.get(qid, [])
        for tid in c_list:
            t_name = target_name_dict[tid]
            t_addr = target_addr_dict[tid]
            sc = scores_dict.get((qid, tid), 0.0)
            feat = extract_pair_features(q_name, q_addr, t_name, t_addr, sc)
            batch_feats.append(feat)
            batch_pairs.append((qid, tid))
    t_feat = time.time() - t0
    
    print(f"500 queries: Blocking={t_block:.3f}s ({500/t_block:.1f} q/s), Feats ({len(batch_pairs)} pairs)={t_feat:.3f}s ({len(batch_pairs)/max(t_feat, 0.001):.1f} pairs/s)")

run_bench()
