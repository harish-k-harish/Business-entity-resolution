import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load sample
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=5000)
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=20000)

names1 = s1['business_name'].fillna('').astype(str).tolist()
names2 = s2['business_name'].fillna('').astype(str).tolist()

t0 = time.time()
vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), min_df=2, max_features=50000)
X2 = vec.fit_transform(names2)
X1 = vec.transform(names1)
print(f"TF-IDF fitted & transformed in {time.time()-t0:.2f}s. Shape: {X2.shape}")

t0 = time.time()
# Batch dot product
batch_size = 500
top_k = 10
candidates = []
for i in range(0, len(names1), batch_size):
    sims = X1[i:i+batch_size].dot(X2.T).toarray()
    top_idx = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
    # sort top_k
    for row_i, idxs in enumerate(top_idx):
        sorted_idxs = idxs[np.argsort(-sims[row_i, idxs])]
        candidates.append(sorted_idxs)
print(f"5,000 queries against 20,000 targets searched in {time.time()-t0:.2f}s ({len(names1)/(time.time()-t0):.1f} q/s)")
