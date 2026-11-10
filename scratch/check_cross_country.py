import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

# Load 10,000 GT rows
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=20000)
gt = gt.dropna()

s1_to_matches = {}
all_mids = set()
for _, r in gt.iterrows():
    mids = [m.strip() for m in str(r['matched_entity_ids']).split(',') if m.strip()]
    if mids:
        s1_to_matches[r['source1_entity_id']] = mids
        all_mids.update(mids)

s1_countries = {}
for chunk in pd.read_csv('dataset/train/train_source1.tsv', sep='\t', chunksize=200000):
    hits = chunk[chunk['entity_id'].isin(s1_to_matches.keys())]
    for _, r in hits.iterrows():
        s1_countries[r['entity_id']] = r['country']
    if len(s1_countries) >= len(s1_to_matches):
        break

match_countries = {}
for s_name, path in [('S2', 'dataset/train/train_source2.tsv'), ('S3', 'dataset/train/train_source3.tsv')]:
    for chunk in pd.read_csv(path, sep='\t', chunksize=200000):
        hits = chunk[chunk['entity_id'].isin(all_mids)]
        for _, r in hits.iterrows():
            match_countries[r['entity_id']] = r['country']

cross_country = 0
total_checked = 0
for s1_id, mids in s1_to_matches.items():
    c1 = s1_countries.get(s1_id)
    if not c1:
        continue
    for mid in mids:
        if mid in match_countries:
            total_checked += 1
            if match_countries[mid] != c1:
                cross_country += 1

print(f"Checked {total_checked} pairs. Cross country count: {cross_country} ({(cross_country/max(1, total_checked))*100:.4f}%)")
