import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

# Inspect train ground truth
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=50000)
empty_count = gt['matched_entity_ids'].isna().sum() + (gt['matched_entity_ids'] == '').sum()
print(f'Train GT first 50k rows: {empty_count} singletons ({empty_count/len(gt)*100:.2f}%)')

# Check countries in train vs test
s1_train = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=50000)
print('Train S1 countries:\n', s1_train['country'].value_counts())

s1_test = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', nrows=50000)
print('Test S1 countries:\n', s1_test['country'].value_counts())

# Read first 100 rows of S1 train and their matched IDs
s1_subset = s1_train.head(100).set_index('entity_id')
gt_subset = gt[gt['source1_entity_id'].isin(s1_subset.index)].dropna()

target_ids = set()
for mids in gt_subset['matched_entity_ids']:
    for mid in mids.split(','):
        if mid.strip():
            target_ids.add(mid.strip())

print(f'Looking for {len(target_ids)} target matched IDs...')

# Search in S2 and S3 train
s2_matches = {}
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['entity_id'].isin(target_ids)]
    for _, r in hits.iterrows():
        s2_matches[r['entity_id']] = r.to_dict()
    if len(s2_matches) >= len(target_ids):
        break

s3_matches = {}
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['entity_id'].isin(target_ids)]
    for _, r in hits.iterrows():
        s3_matches[r['entity_id']] = r.to_dict()
    if len(s3_matches) >= len(target_ids):
        break

print(f'Found {len(s2_matches)} in S2 and {len(s3_matches)} in S3')

# Print 10 detailed examples
count = 0
for _, r in gt_subset.iterrows():
    s1_id = r['source1_entity_id']
    s1_row = s1_subset.loc[s1_id]
    mids = [m.strip() for m in r['matched_entity_ids'].split(',') if m.strip()]
    found_any = any(m in s2_matches or m in s3_matches for m in mids)
    if not found_any:
        continue
    print('='*70)
    print(f"S1 ID: {s1_id} | Country: {s1_row['country']}")
    print(f"  Name: {s1_row['business_name']}")
    print(f"  Address: {s1_row['business_address']}")
    for mid in mids:
        if mid in s2_matches:
            m = s2_matches[mid]
            print(f"  -> MATCHED S2: {mid} | Country: {m['country']}")
            print(f"     Name: {m['business_name']}")
            print(f"     Address: {m['business_address']}")
        elif mid in s3_matches:
            m = s3_matches[mid]
            print(f"  -> MATCHED S3: {mid} | Country: {m['country']}")
            print(f"     Name: {m['business_name']}")
            print(f"     Address: {m['business_address']}")
    count += 1
    if count >= 8:
        break
