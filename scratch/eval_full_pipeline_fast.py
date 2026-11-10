import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz
import lightgbm as lgb
from test_fast_norm import fast_normalize

print("=" * 60)
print("FAST END-TO-END PIPELINE VALIDATION")
print("=" * 60)

# Step 1: Load training and ground truth sample
t0 = time.time()
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=30000)

# Sample 5,000 S1 entities
SAMPLE_SIZE = 5000
sample_s1_ids = gt_df['source1_entity_id'].iloc[:SAMPLE_SIZE].tolist()
s1_id_set = set(sample_s1_ids)

# Ground truth map
gt_dict = {}
all_matched_ids = set()
for _, r in gt_df[gt_df['source1_entity_id'].isin(s1_id_set)].iterrows():
    mids = [m.strip() for m in str(r['matched_entity_ids']).split(',') if m.strip() and m.strip() != 'nan']
    gt_dict[r['source1_entity_id']] = set(mids)
    all_matched_ids.update(mids)

# Load S1 entities
s1_rows = []
for chunk in pd.read_csv('dataset/train/train_source1.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['entity_id'].isin(s1_id_set)]
    s1_rows.append(hits)
    if sum(len(c) for c in s1_rows) >= len(sample_s1_ids):
        break
df_s1 = pd.concat(s1_rows, ignore_index=True).drop_duplicates('entity_id')

# Load target pool: background records (50k each from S2, S3) plus all true matches
needed_s2 = [m for m in all_matched_ids if m.startswith('S2-')]
needed_s3 = [m for m in all_matched_ids if m.startswith('S3-')]

s2_rows = [pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=50000)]
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['entity_id'].isin(needed_s2)]
    if len(hits) > 0:
        s2_rows.append(hits)
df_s2 = pd.concat(s2_rows, ignore_index=True).drop_duplicates('entity_id')

s3_rows = [pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=50000)]
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['entity_id'].isin(needed_s3)]
    if len(hits) > 0:
        s3_rows.append(hits)
df_s3 = pd.concat(s3_rows, ignore_index=True).drop_duplicates('entity_id')

df_targets = pd.concat([df_s2, df_s3], ignore_index=True).drop_duplicates('entity_id')
print(f"Loaded {len(df_s1)} S1 queries, target pool {len(df_targets)} records (S2: {len(df_s2)}, S3: {len(df_s3)}) in {time.time()-t0:.2f}s", flush=True)

# Step 2: Normalization
t0 = time.time()
df_s1['norm_name'] = df_s1['business_name'].apply(lambda x: fast_normalize(x, is_address=False))
df_s1['norm_addr'] = df_s1['business_address'].apply(lambda x: fast_normalize(x, is_address=True))
df_targets['norm_name'] = df_targets['business_name'].apply(lambda x: fast_normalize(x, is_address=False))
df_targets['norm_addr'] = df_targets['business_address'].apply(lambda x: fast_normalize(x, is_address=True))
print(f"Fast normalization completed in {time.time()-t0:.2f}s", flush=True)

# Step 3: Train / Val split (70 / 30)
np.random.seed(42)
perm = np.random.permutation(len(df_s1))
split_idx = int(0.7 * len(df_s1))
train_s1 = df_s1.iloc[perm[:split_idx]].reset_index(drop=True)
val_s1 = df_s1.iloc[perm[split_idx:]].reset_index(drop=True)
print(f"Train S1: {len(train_s1)}, Val S1: {len(val_s1)}", flush=True)

# Step 4: Candidate Generation via Country-Partitioned TF-IDF
TOP_K_CANDS = 25

def generate_candidates_by_country(queries_df, targets_df, top_k=TOP_K_CANDS):
    all_cands = {}
    name_scores = {}
    addr_scores = {}
    comb_scores = {}
    
    for country in queries_df['country'].unique():
        sub_q = queries_df[queries_df['country'] == country]
        sub_t = targets_df[targets_df['country'] == country]
        
        if len(sub_q) == 0 or len(sub_t) == 0:
            for qid in sub_q['entity_id']:
                all_cands[qid] = []
            continue
            
        t_ids = sub_t['entity_id'].values
        
        # TF-IDF on normalized name (char 3-gram)
        v_name = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=2, max_features=40000, sublinear_tf=True)
        X_tgt_name = v_name.fit_transform(sub_t['norm_name'])
        X_q_name = v_name.transform(sub_q['norm_name'])
        
        # TF-IDF on normalized address (char 3-gram)
        v_addr = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=2, max_features=40000, sublinear_tf=True)
        X_tgt_addr = v_addr.fit_transform(sub_t['norm_addr'])
        X_q_addr = v_addr.transform(sub_q['norm_addr'])
        
        # Compute dot products
        sim_name_csr = X_q_name.dot(X_tgt_name.T).tocsr()
        sim_addr_csr = X_q_addr.dot(X_tgt_addr.T).tocsr()
        
        # Combined score: 1.0 * name + 0.6 * addr
        sim_comb_csr = (sim_name_csr + 0.6 * sim_addr_csr).tocsr()
        
        for i, qid in enumerate(sub_q['entity_id']):
            r_start = sim_comb_csr.indptr[i]
            r_end = sim_comb_csr.indptr[i+1]
            data = sim_comb_csr.data[r_start:r_end]
            indices = sim_comb_csr.indices[r_start:r_end]
            
            if len(data) == 0:
                all_cands[qid] = []
                continue
            elif len(data) <= top_k:
                top_part = np.argsort(-data)
            else:
                part = np.argpartition(-data, top_k)[:top_k]
                top_part = part[np.argsort(-data[part])]
                
            selected_t_indices = indices[top_part]
            selected_t_ids = t_ids[selected_t_indices]
            all_cands[qid] = selected_t_ids.tolist()
            
            # Record individual scores for feature extractor
            for rank, tid in enumerate(selected_t_ids):
                comb_scores[(qid, tid)] = float(data[top_part[rank]])
                
    return all_cands, comb_scores

print("\n--- Running Candidate Generation (Blocking) ---", flush=True)
t0 = time.time()
train_cands, train_comb_scores = generate_candidates_by_country(train_s1, df_targets)
val_cands, val_comb_scores = generate_candidates_by_country(val_s1, df_targets)
print(f"Candidates generated in {time.time()-t0:.2f}s", flush=True)

# Evaluate Validation Blocking Recall
target_id_set = set(df_targets['entity_id'])
val_measurable_gt = {}
for s1_id in val_s1['entity_id']:
    mids = gt_dict.get(s1_id, set()) & target_id_set
    if mids:
        val_measurable_gt[s1_id] = mids

total_val_targets = sum(len(v) for v in val_measurable_gt.values())
hits = sum(len(set(val_cands.get(s1_id, [])) & t_set) for s1_id, t_set in val_measurable_gt.items())
recall = (hits / total_val_targets * 100) if total_val_targets > 0 else 0
print(f"Validation Candidate Blocking Recall (top-{TOP_K_CANDS}): {recall:.2f}% ({hits}/{total_val_targets})", flush=True)

# Step 5: Feature Extraction
print("\n--- Feature Extraction ---", flush=True)
target_lookup = df_targets.set_index('entity_id')[['norm_name', 'norm_addr']].to_dict('index')
num_regex = re.compile(r'\b\d+\b')

def extract_pairwise_features(s1_df, cands_map, comb_scores, is_train=True):
    s1_dict = s1_df.set_index('entity_id')[['norm_name', 'norm_addr']].to_dict('index')
    
    rows = []
    labels = []
    pairs = []
    
    for s1_id, c_list in cands_map.items():
        s1_info = s1_dict[s1_id]
        n1 = s1_info['norm_name']
        a1 = s1_info['norm_addr']
        s1_nums = set(num_regex.findall(a1))
        s1_toks = set(a1.split())
        true_mids = gt_dict.get(s1_id, set())
        
        for tid in c_list:
            if tid not in target_lookup:
                continue
            t_info = target_lookup[tid]
            n2 = t_info['norm_name']
            a2 = t_info['norm_addr']
            
            # String similarities
            jw_name = JaroWinkler.similarity(n1, n2)
            tsr_name = fuzz.token_sort_ratio(n1, n2) / 100.0
            tset_name = fuzz.token_set_ratio(n1, n2) / 100.0
            ratio_name = fuzz.ratio(n1, n2) / 100.0
            
            tsr_addr = fuzz.token_sort_ratio(a1, a2) / 100.0
            tset_addr = fuzz.token_set_ratio(a1, a2) / 100.0
            
            # Address token Jaccard
            t2_toks = set(a2.split())
            union_len = len(s1_toks | t2_toks)
            jaccard_addr = (len(s1_toks & t2_toks) / union_len) if union_len > 0 else 0.0
            
            # Numeric sequence exact match & overlap
            t2_nums = set(num_regex.findall(a2))
            if s1_nums and t2_nums:
                num_exact = 1.0 if s1_nums == t2_nums else 0.0
                num_overlap = len(s1_nums & t2_nums) / len(s1_nums | t2_nums)
            elif not s1_nums and not t2_nums:
                num_exact = 0.5
                num_overlap = 0.5
            else:
                num_exact = 0.0
                num_overlap = 0.0
                
            name_len_diff = abs(len(n1) - len(n2)) / max(1, max(len(n1), len(n2)))
            comb_sc = comb_scores.get((s1_id, tid), 0.0)
            
            feat = [
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
                comb_sc
            ]
            rows.append(feat)
            pairs.append((s1_id, tid))
            if is_train:
                labels.append(1 if tid in true_mids else 0)
                
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32) if is_train else None
    return X, y, pairs

t0 = time.time()
X_train, y_train, train_pairs = extract_pairwise_features(train_s1, train_cands, train_comb_scores, is_train=True)
X_val, y_val, val_pairs = extract_pairwise_features(val_s1, val_cands, val_comb_scores, is_train=True)
dt = time.time() - t0
print(f"Features extracted: X_train {X_train.shape} (pos rate: {y_train.mean():.4f}), X_val {X_val.shape} in {dt:.2f}s ({(len(X_train)+len(X_val))/dt:.1f} pairs/sec)", flush=True)

# Step 6: Train LightGBM Classifier
print("\n--- Training LightGBM Matcher ---", flush=True)
feature_names = [
    'jw_name', 'tsr_name', 'tset_name', 'ratio_name',
    'tsr_addr', 'tset_addr', 'jaccard_addr', 'num_exact',
    'num_overlap', 'name_len_diff', 'comb_sparse_score'
]

clf = lgb.LGBMClassifier(
    n_estimators=250,
    learning_rate=0.06,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf.fit(X_train, y_train)

print("\n--- Feature Importances ---")
for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:20s}: {imp}")

# Step 7: Macro F_0.5 Evaluation & Threshold Tuning
val_probs = clf.predict_proba(X_val)[:, 1]

def evaluate_macro_f05(preds_map, ground_truth_map, eval_s1_ids):
    scores = []
    for s1_id in eval_s1_ids:
        preds = set(preds_map.get(s1_id, []))
        truth = set(ground_truth_map.get(s1_id, []))
        
        # Singleton evaluation
        if len(truth) == 0:
            scores.append(1.0 if len(preds) == 0 else 0.0)
        else:
            if len(preds) == 0:
                scores.append(0.0)
            else:
                tp = len(preds & truth)
                if tp == 0:
                    scores.append(0.0)
                else:
                    prec = tp / len(preds)
                    rec = tp / len(truth)
                    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                    scores.append(f05)
    return float(np.mean(scores))

val_ids = val_s1['entity_id'].tolist()
best_thresh = 0.5
best_score = 0.0

print("\n--- Threshold Optimization for Macro F_0.5 ---", flush=True)
for thresh in np.arange(0.40, 0.96, 0.04):
    preds = {s1_id: [] for s1_id in val_ids}
    for (s1_id, tid), prob in zip(val_pairs, val_probs):
        if prob >= thresh:
            preds[s1_id].append(tid)
            
    score = evaluate_macro_f05(preds, gt_dict, val_ids)
    print(f"  Threshold {thresh:.2f} -> Macro F_0.5: {score:.5f}", flush=True)
    if score > best_score:
        best_score = score
        best_thresh = thresh

print(f"\n========================================================")
print(f"FINAL RESULT: Optimal Threshold: {best_thresh:.2f} | Macro F_0.5: {best_score:.5f}")
print(f"========================================================")
