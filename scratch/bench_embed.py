import time
from sentence_transformers import SentenceTransformer

print("Loading model paraphrase-multilingual-MiniLM-L12-v2...")
t0 = time.time()
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print(f"Model loaded in {time.time()-t0:.2f}s")

texts = ["1795 Westchester Drive, High Point, NC Orelee's Barbershop"] * 100
t0 = time.time()
embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
elapsed = time.time() - t0
print(f"100 texts encoded in {elapsed:.2f}s ({100/elapsed:.1f} texts/sec)")
