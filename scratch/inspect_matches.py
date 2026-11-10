import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=10)
print('GT head IDs:', gt['source1_entity_id'].tolist())

s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=10)
print('S1 head IDs:', s1['entity_id'].tolist())

# Look up S1 for the first GT row
gt_first = gt.iloc[0]
first_s1_id = gt_first['source1_entity_id']
first_matches = [m.strip() for m in str(gt_first['matched_entity_ids']).split(',') if m.strip()]
print('First GT:', first_s1_id, 'matches:', first_matches)

# Find first_s1_id in s1
for chunk in pd.read_csv('dataset/train/train_source1.tsv', sep='\t', chunksize=100000):
    hits = chunk[chunk['entity_id'] == first_s1_id]
    if len(hits) > 0:
        print('Found S1:', hits.iloc[0].to_dict())
        break

# Find matches in S2/S3
for m in first_matches:
    if m.startswith('S2-'):
        for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=200000):
            hits = chunk[chunk['entity_id'] == m]
            if len(hits) > 0:
                print('Found S2 match:', hits.iloc[0].to_dict())
                break
    elif m.startswith('S3-'):
        for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=200000):
            hits = chunk[chunk['entity_id'] == m]
            if len(hits) > 0:
                print('Found S3 match:', hits.iloc[0].to_dict())
                break
