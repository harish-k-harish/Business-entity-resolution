import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import time
from test_norm import normalize_text

print("Loading train ground truth sample...")
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=20000)
# Filter to entities that have matches
gt_with_matches = gt.dropna().copy()
gt_with_matches = gt_with_matches[gt_with_matches['matched_entity_ids'].str.strip() != '']
sample_s1_ids = set(gt_with_matches.head(1000)['source1_entity_id'])

# Load S1 sample
s1_rows = []
for chunk in pd.read_csv('dataset/train/train_source1.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['entity_id'].isin(sample_s1_ids)]
    s1_rows.append(hits)
    if sum(len(c) for c in s1_rows) >= len(sample_s1_ids):
        break
df_s1 = pd.concat(s1_rows, ignore_index=True)

# Build map of true matches for these S1
gt_map = {}
all_true_mids = set()
for _, r in gt_with_matches[gt_with_matches['source1_entity_id'].isin(df_s1['entity_id'])].iterrows():
    mids = [m.strip() for m in r['matched_entity_ids'].split(',') if m.strip()]
    gt_map[r['source1_entity_id']] = set(mids)
    all_true_mids.update(mids)

print(f"Sample: {len(df_s1)} S1 entities with {len(all_true_mids)} true match targets across S2/S3")

# Load candidate pool from S2 and S3 for same countries
countries = df_s1['country'].unique()
print(f"Countries in sample: {countries}")

# For testing, let's load first 100k S2 and 100k S3 records to ensure all true_mids in that slice are found
s2_rows = []
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['country'].isin(countries)]
    s2_rows.append(hits)
    if sum(len(c) for c in s2_rows) >= 100000:
        break
df_s2 = pd.concat(s2_rows, ignore_index=True)

s3_rows = []
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['country'].isin(countries)]
    s3_rows.append(hits)
    if sum(len(c) for c in s3_rows) >= 100000:
        break
df_s3 = pd.concat(s3_rows, ignore_index=True)

# Combine S2 and S3
df_targets = pd.concat([df_s2, df_s3], ignore_index=True)
target_id_set = set(df_targets['entity_id'])

# How many true matches exist in our pool?
measurable_gt = {}
for s1_id, t_set in gt_map.items():
    in_pool = t_set & target_id_set
    if in_pool:
        measurable_gt[s1_id] = in_pool

print(f"Entities with true matches in our pool: {len(measurable_gt)}, total measurable targets: {sum(len(v) for v in measurable_gt.values())}")

# Normalize names
df_s1['norm_name'] = df_s1['business_name'].apply(lambda x: normalize_text(x, is_address=False))
df_targets['norm_name'] = df_targets['business_name'].apply(lambda x: normalize_text(x, is_address=False))

# Test TF-IDF char 3-gram retrieval
t0 = time.time()
tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=2, max_features=50000)
X_targets = tfidf.fit_transform(df_targets['norm_name'])
X_s1 = tfidf.transform(df_s1['norm_name'])

top_k = 30
target_ids = df_targets['entity_id'].values

# Query batch
hits_found = 0
total_targets = sum(len(v) for v in measurable_gt.values())

for i, s1_id in enumerate(df_s1['entity_id']):
    if s1_id not in measurable_gt:
        continue
    row = X_s1[i].dot(X_targets.T).toarray().ravel()
    top_indices = np.argpartition(-row, top_k)[:top_k]
    cand_ids = set(target_ids[top_indices])
    true_ids = measurable_gt[s1_id]
    hits_found += len(true_ids & cand_ids)

recall = hits_found / total_targets if total_targets else 0
print(f"Char 3-gram TF-IDF top-{top_k} Recall: {recall*100:.2f}% ({hits_found}/{total_targets}) in {time.time()-t0:.2f}s")
