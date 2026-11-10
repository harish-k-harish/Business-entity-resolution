import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

for split in ['test']:
    for f in ['test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv']:
        path = f'dataset/{split}/{f}'
        print(f"=== {f} ===")
        # Read in chunks to get country distribution and check missing values
        country_counts = {}
        total = 0
        null_names = 0
        null_addrs = 0
        for chunk in pd.read_csv(path, sep='\t', chunksize=500000):
            total += len(chunk)
            null_names += chunk['business_name'].isna().sum()
            null_addrs += chunk['business_address'].isna().sum()
            for c, cnt in chunk['country'].value_counts().items():
                country_counts[c] = country_counts.get(c, 0) + cnt
        print(f"Total: {total}, Null names: {null_names}, Null addrs: {null_addrs}")
        print("Countries:", country_counts)
