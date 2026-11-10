import time
import re
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz

names1 = ["Orelee's Barbershop", "Vision Partners Corp", "Zephay Labs Inc"] * 10000
names2 = ["Orelee Barbershop LLC", "Vision Partners", "Zephay Labs Incorporated"] * 10000

addrs1 = ["1795 Westchester Drive, High Point, NC", "1064 Newton Rd, Unit 11", "2621 Cotten Road, Tyler, TX"] * 10000
addrs2 = ["1795 Westchester Dr, High Point", "1064 Newton Road Suite 11", "2621 Cotten Rd, Tyler"] * 10000

num_regex = re.compile(r'\b\d+\b')

t0 = time.time()
for n1, n2, a1, a2 in zip(names1, names2, addrs1, addrs2):
    # Jaro-Winkler
    jw_name = JaroWinkler.similarity(n1, n2)
    # Token sort ratio
    tsr_name = fuzz.token_sort_ratio(n1, n2) / 100.0
    tsr_addr = fuzz.token_sort_ratio(a1, a2) / 100.0
    # Jaccard on address tokens
    s1 = set(a1.lower().split())
    s2 = set(a2.lower().split())
    jaccard_addr = len(s1 & s2) / max(1, len(s1 | s2))
    # Numeric sequence exact match
    num1 = set(num_regex.findall(a1))
    num2 = set(num_regex.findall(a2))
    num_match = 1.0 if (num1 and num1 == num2) else 0.0

dt = time.time() - t0
print(f"Computed 30,000 feature vectors in {dt:.2f}s ({len(names1)/dt:.1f} pairs/sec)")
