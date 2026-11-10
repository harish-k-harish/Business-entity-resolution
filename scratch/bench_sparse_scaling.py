import time
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

print("Creating synthetic sparse matrices...")
n_vocab = 30000
n_targets = 200000
n_queries = 20000

# Average 15 non-zero entries per row
X_targets = sparse.random(n_targets, n_vocab, density=15/n_vocab, format='csr', dtype=np.float32)
X_queries = sparse.random(n_queries, n_vocab, density=15/n_vocab, format='csr', dtype=np.float32)

t0 = time.time()
batch_size = 5000
top_k = 30
total_candidates = 0

for i in range(0, n_queries, batch_size):
    q_batch = X_queries[i:i+batch_size]
    # Matrix product
    t_dot = time.time()
    sims = q_batch.dot(X_targets.T) # csr matrix
    
    # Fast top-k selection from sparse matrix
    for r in range(q_batch.shape[0]):
        row_data = sims[r].data
        if len(row_data) > 0:
            row_indices = sims[r].indices
            if len(row_data) <= top_k:
                top_idx = row_indices[np.argsort(-row_data)]
            else:
                part = np.argpartition(-row_data, top_k)[:top_k]
                top_idx = row_indices[part[np.argsort(-row_data[part])]]
            total_candidates += len(top_idx)

dt = time.time() - t0
print(f"Searched {n_queries} queries against {n_targets} targets in {dt:.2f}s ({n_queries/dt:.1f} q/s)")
