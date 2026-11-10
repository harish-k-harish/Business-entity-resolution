import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import re
import time
from test_norm import normalize_text
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from sentence_transformers import SentenceTransformer
import lightgbm as lgb

print("--- Step 1: Loading Training and Ground Truth Data ---")
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=25000)
# Sample 3,000 entities
s1_ids_sample = gt['source1_entity_id'].iloc[:3000].tolist()

# Load S1 records
s1_list = []
for chunk in pd.read_csv('dataset/train/train_source1.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['entity_id'].isin(s1_ids_sample)]
    s1_list.append(hits)
    if sum(len(c) for c in s1_list) >= len(s1_ids_sample):
        break
df_s1 = pd.concat(s1_list, ignore_index=True)

# Build GT dictionary
gt_dict = {}
all_matched_ids = set()
for _, r in gt[gt['source1_entity_id'].isin(df_s1['entity_id'])].iterrows():
    mids = [m.strip() for m in str(r['matched_entity_ids']).split(',') if m.strip() and m.strip() != 'nan']
    gt_dict[r['source1_entity_id']] = set(mids)
    all_matched_ids.update(mids)

print(f"Loaded {len(df_s1)} S1 entities, {len(all_matched_ids)} total matched target IDs.")

# Target records from S2 and S3
needed_s2 = [m for m in all_matched_ids if m.startswith('S2-')]
needed_s3 = [m for m in all_matched_ids if m.startswith('S3-')]

print("Loading target records and background pool...")
# Load background pool plus all ground truth matches
s2_list = [pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=40000)]
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['entity_id'].isin(needed_s2)]
    if len(hits) > 0:
        s2_list.append(hits)
df_s2 = pd.concat(s2_list, ignore_index=True).drop_duplicates('entity_id')

s3_list = [pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=40000)]
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['entity_id'].isin(needed_s3)]
    if len(hits) > 0:
        s3_list.append(hits)
df_s3 = pd.concat(s3_list, ignore_index=True).drop_duplicates('entity_id')

df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
print(f"Target pool: {len(df_targets)} records (S2: {len(df_s2)}, S3: {len(df_s3)})")

# Normalization
df_s1['norm_name'] = df_s1['business_name'].apply(lambda x: normalize_text(x, is_address=False))
df_s1['norm_addr'] = df_s1['business_address'].apply(lambda x: normalize_text(x, is_address=True))
df_s1['combined'] = df_s1['norm_name'] + " " + df_s1['norm_addr']

df_targets['norm_name'] = df_targets['business_name'].apply(lambda x: normalize_text(x, is_address=False))
df_targets['norm_addr'] = df_targets['business_address'].apply(lambda x: normalize_text(x, is_address=True))
df_targets['combined'] = df_targets['norm_name'] + " " + df_targets['norm_addr']

# Train/Val Split (70/30) on S1 entities
np.random.seed(42)
all_s1_indices = np.random.permutation(len(df_s1))
split_idx = int(0.7 * len(df_s1))
train_s1_df = df_s1.iloc[all_s1_indices[:split_idx]].reset_index(drop=True)
val_s1_df = df_s1.iloc[all_s1_indices[split_idx:]].reset_index(drop=True)
print(f"Train S1: {len(train_s1_df)}, Val S1: {len(val_s1_df)}")

# Country partition blocking
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def generate_candidates(s1_df, targets_df, top_k_sparse=20, top_k_dense=20, top_k_merged=30):
    candidates = {}
    sparse_scores = {}
    dense_scores = {}
    
    # Process by country
    for c in s1_df['country'].unique():
        sub_s1 = s1_df[s1_df['country'] == c]
        sub_targets = targets_df[targets_df['country'] == c]
        if len(sub_s1) == 0 or len(sub_targets) == 0:
            for s1_id in sub_s1['entity_id']:
                candidates[s1_id] = []
            continue
        
        target_entity_ids = sub_targets['entity_id'].values
        
        # 1. Sparse TF-IDF on norm_name
        tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=2, max_features=40000)
        X_target = tfidf.fit_transform(sub_targets['norm_name'])
        X_query = tfidf.transform(sub_s1['norm_name'])
        
        k_sp = min(top_k_sparse, len(sub_targets))
        k_dn = min(top_k_dense, len(sub_targets))
        
        # Dense encoding
        emb_query = model.encode(sub_s1['combined'].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        emb_target = model.encode(sub_targets['combined'].tolist(), batch_size=128, normalize_embeddings=True, show_progress_bar=False)
        
        dim = emb_target.shape[1]
        faiss_idx = faiss.IndexFlatIP(dim)
        faiss_idx.add(emb_target.astype(np.float32))
        d_scores, d_indices = faiss_idx.search(emb_query.astype(np.float32), k_dn)
        
        # Batch dot product for sparse
        sims_matrix = X_query.dot(X_target.T)
        
        for i, s1_id in enumerate(sub_s1['entity_id']):
            # Sparse candidates
            row = sims_matrix[i].toarray().ravel()
            sp_top = np.argpartition(-row, k_sp)[:k_sp]
            sp_sorted = sp_top[np.argsort(-row[sp_top])]
            
            cands_dict = {}
            for idx in sp_sorted:
                tid = target_entity_ids[idx]
                cands_dict[tid] = {'sparse': float(row[idx]), 'dense': 0.0}
            
            # Dense candidates
            for rank in range(k_dn):
                t_idx = d_indices[i, rank]
                tid = target_entity_ids[t_idx]
                score = float(d_scores[i, rank])
                if tid not in cands_dict:
                    cands_dict[tid] = {'sparse': 0.0, 'dense': score}
                else:
                    cands_dict[tid]['dense'] = score
            
            # Merge & sort by combined ranking
            sorted_cands = sorted(cands_dict.keys(), key=lambda x: (cands_dict[x]['sparse'] + cands_dict[x]['dense']), reverse=True)[:top_k_merged]
            candidates[s1_id] = sorted_cands
            for tid in sorted_cands:
                sparse_scores[(s1_id, tid)] = cands_dict[tid]['sparse']
                dense_scores[(s1_id, tid)] = cands_dict[tid]['dense']
                
    return candidates, sparse_scores, dense_scores

print("Generating candidates for Train...")
t0 = time.time()
train_cands, train_sp_scores, train_dn_scores = generate_candidates(train_s1_df, df_targets)
print(f"Train candidates generated in {time.time()-t0:.2f}s")

print("Generating candidates for Val...")
t0 = time.time()
val_cands, val_sp_scores, val_dn_scores = generate_candidates(val_s1_df, df_targets)
print(f"Val candidates generated in {time.time()-t0:.2f}s")

# Target lookup dicts
target_lookup = df_targets.set_index('entity_id')[['norm_name', 'norm_addr']].to_dict('index')
num_regex = re.compile(r'\b\d+\b')

def extract_features(s1_df, cands_map, sp_scores, dn_scores, is_train=True):
    s1_dict = s1_df.set_index('entity_id')[['norm_name', 'norm_addr']].to_dict('index')
    
    rows = []
    labels = []
    pair_ids = []
    
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
            
            # Rapidfuzz features
            jw_name = JaroWinkler.similarity(n1, n2)
            tsr_name = fuzz.token_sort_ratio(n1, n2) / 100.0
            tsr_addr = fuzz.token_sort_ratio(a1, a2) / 100.0
            
            # Jaccard on address tokens
            t2_toks = set(a2.split())
            union_len = len(s1_toks | t2_toks)
            jaccard_addr = (len(s1_toks & t2_toks) / union_len) if union_len > 0 else 0.0
            
            # Exact numeric match
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
                
            # Additional features
            tset_name = fuzz.token_set_ratio(n1, n2) / 100.0
            tset_addr = fuzz.token_set_ratio(a1, a2) / 100.0
            name_len_diff = abs(len(n1) - len(n2)) / max(1, max(len(n1), len(n2)))
            
            sp_score = sp_scores.get((s1_id, tid), 0.0)
            dn_score = dn_scores.get((s1_id, tid), 0.0)
            
            feat = [
                jw_name,
                tsr_name,
                tsr_addr,
                jaccard_addr,
                num_exact,
                num_overlap,
                tset_name,
                tset_addr,
                name_len_diff,
                sp_score,
                dn_score
            ]
            rows.append(feat)
            pair_ids.append((s1_id, tid))
            
            if is_train:
                labels.append(1 if tid in true_mids else 0)
                
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32) if is_train else None
    return X, y, pair_ids

print("Extracting features...")
X_train, y_train, train_pairs = extract_features(train_s1_df, train_cands, train_sp_scores, train_dn_scores, is_train=True)
X_val, y_val, val_pairs = extract_features(val_s1_df, val_cands, val_sp_scores, val_dn_scores, is_train=True)
print(f"X_train shape: {X_train.shape}, positive rate: {y_train.mean():.4f}")
print(f"X_val shape: {X_val.shape}, positive rate: {y_val.mean():.4f}")

# Train LightGBM
print("Training LightGBM...")
feature_names = [
    'jw_name', 'tsr_name', 'tsr_addr', 'jaccard_addr',
    'num_exact', 'num_overlap', 'tset_name', 'tset_addr',
    'name_len_diff', 'sparse_score', 'dense_score'
]

clf = lgb.LGBMClassifier(
    n_estimators=150,
    learning_rate=0.08,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf.fit(X_train, y_train)

# Feature importances
print("\n--- Feature Importances ---")
for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:15s}: {imp}")

# Predict probabilities on validation
val_probs = clf.predict_proba(X_val)[:, 1]

# Macro F_0.5 evaluation function
def calc_macro_f05(preds_dict, ground_truth_dict, all_eval_ids):
    f_scores = []
    for s1_id in all_eval_ids:
        preds = set(preds_dict.get(s1_id, []))
        truth = set(ground_truth_dict.get(s1_id, []))
        
        # Singleton case
        if len(truth) == 0:
            if len(preds) == 0:
                f_scores.append(1.0)
            else:
                f_scores.append(0.0)
        else:
            if len(preds) == 0:
                f_scores.append(0.0)
            else:
                tp = len(preds & truth)
                if tp == 0:
                    f_scores.append(0.0)
                else:
                    prec = tp / len(preds)
                    rec = tp / len(truth)
                    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                    f_scores.append(f05)
    return np.mean(f_scores)

# Grid search threshold
val_ids = val_s1_df['entity_id'].tolist()
best_thresh = 0.5
best_score = 0.0

print("\n--- Grid Search for Probability Threshold (Macro F_0.5) ---")
for thresh in np.arange(0.50, 0.98, 0.03):
    preds_dict = {s1_id: [] for s1_id in val_ids}
    for (s1_id, tid), prob in zip(val_pairs, val_probs):
        if prob >= thresh:
            preds_dict[s1_id].append(tid)
            
    score = calc_macro_f05(preds_dict, gt_dict, val_ids)
    print(f"Threshold: {thresh:.2f} -> Macro F_0.5: {score:.5f}")
    if score > best_score:
        best_score = score
        best_thresh = thresh

print(f"\n>>> Optimal Threshold: {best_thresh:.2f} with Macro F_0.5: {best_score:.5f} <<<")
