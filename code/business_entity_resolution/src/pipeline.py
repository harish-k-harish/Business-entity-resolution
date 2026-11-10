import os
import sys
# Ensure src directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import gc
import time
import argparse
import pickle
import numpy as np
import pandas as pd

from normalize import normalize_text
from blocking import CountrySparseBlocker
from features import extract_pair_features, FEATURE_NAMES
from model import load_matcher_model

def run_pipeline(
    test_dir: str = 'dataset/test',
    train_dir: str = 'dataset/train',
    output_dir: str = 'output',
    model_path: str = 'code/business_entity_resolution/models/matcher_model.pkl',
    meta_path: str = 'code/business_entity_resolution/models/model_meta.pkl',
    threshold: float = None,
    top_k: int = 20,
    chunk_size: int = 150000
):
    print("=" * 65)
    print("BUSINESS ENTITY RESOLUTION — TEST INFERENCE PIPELINE")
    print("=" * 65)
    t_start = time.time()
    
    # 1. Load model and metadata
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Training model first...")
        from train import train_pipeline
        train_pipeline(train_dir=train_dir, model_output_path=model_path, meta_output_path=meta_path)
        
    print(f"Loading trained matcher model from {model_path}...")
    clf = load_matcher_model(model_path)
    
    if threshold is None:
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            threshold = meta.get('optimal_threshold', 0.68)
            print(f"Loaded optimal threshold from metadata: {threshold:.2f}")
        else:
            threshold = 0.68
            print(f"Using default optimal threshold: {threshold:.2f}")
    else:
        print(f"Using specified threshold: {threshold:.2f}")
        
    os.makedirs(output_dir, exist_ok=True)
    matching_out_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_out_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    
    # Open output files and write headers
    f_match = open(matching_out_path, 'w', encoding='utf-8')
    f_cand = open(candidate_out_path, 'w', encoding='utf-8')
    
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    # 2. Inspect test source1 to identify all countries and entity counts
    s1_test_path = os.path.join(test_dir, 'test_source1.tsv')
    s2_test_path = os.path.join(test_dir, 'test_source2.tsv')
    s3_test_path = os.path.join(test_dir, 'test_source3.tsv')
    
    print(f"Scanning test entities in {s1_test_path}...")
    country_to_s1_indices = {}
    total_s1_records = 0
    
    # We read test_source1 in chunks to record country partition
    # Note: to preserve memory, we process country by country
    for chunk in pd.read_csv(s1_test_path, sep='\t', usecols=['entity_id', 'country'], chunksize=200000):
        total_s1_records += len(chunk)
        for country, group in chunk.groupby('country'):
            if country not in country_to_s1_indices:
                country_to_s1_indices[country] = 0
            country_to_s1_indices[country] += len(group)
            
    print(f"Found {total_s1_records:,} total test S1 entities across countries:")
    for c, cnt in country_to_s1_indices.items():
        print(f"  - {c}: {cnt:,} entities")
        
    total_processed_s1 = 0
    total_matched_pairs = 0
    total_candidate_pairs = 0
    
    # Process country by country
    for country in sorted(country_to_s1_indices.keys()):
        c_time = time.time()
        print(f"\n>>> Processing Country: {country} ({country_to_s1_indices[country]:,} entities) <<<")
        
        # Load targets for this country from S2 and S3
        print(f"Loading {country} target records from test_source2 and test_source3...")
        tgt_list = []
        for p in [s2_test_path, s3_test_path]:
            if not os.path.exists(p):
                continue
            for chunk in pd.read_csv(p, sep='\t', chunksize=200000):
                c_hits = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
                if len(c_hits) > 0:
                    tgt_list.append(c_hits)
                    
        if len(tgt_list) == 0:
            print(f"Warning: No target records found for country {country}!")
            # Stream empty matches for all S1 entities of this country
            for chunk in pd.read_csv(s1_test_path, sep='\t', chunksize=200000):
                c_hits = chunk[chunk['country'] == country]
                for s1_id in c_hits['entity_id']:
                    f_match.write(f"{s1_id}\t\n")
                    f_cand.write(f"{s1_id}\t\n")
                    total_processed_s1 += 1
            continue
            
        df_country_targets = pd.concat(tgt_list, ignore_index=True).drop_duplicates('entity_id')
        del tgt_list
        gc.collect()
        
        num_targets = len(df_country_targets)
        print(f"Normalizing {num_targets:,} target records for {country}...")
        df_country_targets['norm_name'] = df_country_targets['business_name'].apply(lambda x: normalize_text(x, is_address=False))
        df_country_targets['norm_addr'] = df_country_targets['business_address'].apply(lambda x: normalize_text(x, is_address=True))
        
        target_ids = df_country_targets['entity_id'].values
        # Create fast, memory-compact lookup dicts for pairwise features
        target_name_dict = dict(zip(target_ids, df_country_targets['norm_name']))
        target_addr_dict = dict(zip(target_ids, df_country_targets['norm_addr']))
        
        # Fit blocker
        print(f"Fitting sparse TF-IDF blocker on {country} targets...")
        blocker = CountrySparseBlocker(max_features=45000, top_k=top_k, addr_weight=0.6)
        blocker.fit_targets(
            target_ids=target_ids,
            target_norm_names=df_country_targets['norm_name'].tolist(),
            target_norm_addrs=df_country_targets['norm_addr'].tolist()
        )
        
        # Free targets DataFrame to minimize RAM
        del df_country_targets
        gc.collect()
        
        # Stream S1 queries for this country
        print(f"Streaming and scoring {country} queries in batches of {chunk_size:,}...")
        country_s1_done = 0
        
        for chunk in pd.read_csv(s1_test_path, sep='\t', chunksize=chunk_size):
            sub_s1 = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
            if len(sub_s1) == 0:
                continue
                
            sub_s1 = sub_s1.copy()
            sub_s1['norm_name'] = sub_s1['business_name'].apply(lambda x: normalize_text(x, is_address=False))
            sub_s1['norm_addr'] = sub_s1['business_address'].apply(lambda x: normalize_text(x, is_address=True))
            
            s1_ids = sub_s1['entity_id'].tolist()
            s1_names = sub_s1['norm_name'].tolist()
            s1_addrs = sub_s1['norm_addr'].tolist()
            
            # Query blocker
            cands_dict, scores_dict = blocker.query_batch(s1_ids, s1_names, s1_addrs)
            
            # Collect all candidate pairs for feature extraction in batch
            batch_pairs = []
            batch_feats = []
            
            for qid, q_name, q_addr in zip(s1_ids, s1_names, s1_addrs):
                c_list = cands_dict.get(qid, [])
                for tid in c_list:
                    if tid not in target_name_dict:
                        continue
                    t_name = target_name_dict[tid]
                    t_addr = target_addr_dict[tid]
                    sc = scores_dict.get((qid, tid), 0.0)
                    feat = extract_pair_features(q_name, q_addr, t_name, t_addr, sc)
                    batch_feats.append(feat)
                    batch_pairs.append((qid, tid))
                    
            # Predict match probabilities for candidate pairs
            matched_pairs_set = set()
            if len(batch_feats) > 0:
                X_batch = np.array(batch_feats, dtype=np.float32)
                probs = clf.predict_proba(X_batch)[:, 1]
                for (qid, tid), p in zip(batch_pairs, probs):
                    if p >= threshold:
                        matched_pairs_set.add((qid, tid))
                        
            # Write results for each S1 query
            for qid in s1_ids:
                c_list = cands_dict.get(qid, [])
                m_list = [tid for tid in c_list if (qid, tid) in matched_pairs_set]
                
                # Format TSV lines
                cand_str = ",".join(c_list)
                match_str = ",".join(m_list)
                
                f_cand.write(f"{qid}\t{cand_str}\n")
                f_match.write(f"{qid}\t{match_str}\n")
                
                total_candidate_pairs += len(c_list)
                total_matched_pairs += len(m_list)
                
            country_s1_done += len(s1_ids)
            total_processed_s1 += len(s1_ids)
            f_cand.flush()
            f_match.flush()
            
            print(f"  Processed {country_s1_done:,}/{country_to_s1_indices[country]:,} queries ({country_s1_done/country_to_s1_indices[country]*100:.1f}%)", flush=True)
            
        print(f"\nCountry {country} completed in {time.time()-c_time:.2f}s", flush=True)
        del blocker, target_name_dict, target_addr_dict, target_ids
        gc.collect()
        
    f_match.close()
    f_cand.close()
    
    print("\n" + "=" * 65)
    print(f"PIPELINE COMPLETE in {time.time()-t_start:.2f}s")
    print(f"  Total S1 Entities Processed: {total_processed_s1:,} / {total_s1_records:,}")
    print(f"  Total Candidates Generated : {total_candidate_pairs:,}")
    print(f"  Total Matches Predicted    : {total_matched_pairs:,}")
    print(f"  Output files:")
    print(f"    - {matching_out_path}")
    print(f"    - {candidate_out_path}")
    print("=" * 65)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Business Entity Resolution Pipeline")
    parser.add_argument('--test-dir', default='dataset/test', help="Directory containing test TSVs")
    parser.add_argument('--train-dir', default='dataset/train', help="Directory containing train TSVs")
    parser.add_argument('--output-dir', default='output', help="Directory to save output TSVs")
    parser.add_argument('--model-path', default='code/business_entity_resolution/models/matcher_model.pkl')
    parser.add_argument('--meta-path', default='code/business_entity_resolution/models/model_meta.pkl')
    parser.add_argument('--threshold', type=float, default=None, help="Probability threshold override")
    parser.add_argument('--top-k', type=int, default=20, help="Top-K candidates per entity")
    parser.add_argument('--chunk-size', type=int, default=150000, help="Query chunk size")
    args = parser.parse_args()
    
    run_pipeline(
        test_dir=args.test_dir,
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        meta_path=args.meta_path,
        threshold=args.threshold,
        top_k=args.top_k,
        chunk_size=args.chunk_size
    )
