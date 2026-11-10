import time
import torch
from sentence_transformers import SentenceTransformer

torch.set_num_threads(12)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

texts = ["1795 Westchester Drive, High Point, NC Orelee's Barbershop"] * 1000

for bs in [64, 128, 256, 512]:
    t0 = time.time()
    emb = model.encode(texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    dt = time.time() - t0
    print(f"Batch size {bs}: {len(texts)/dt:.1f} texts/sec (total {dt:.2f}s)")
