import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from collections import Counter
import re

def get_tokens(text_series, top_n=40):
    words = []
    for t in text_series.dropna():
        # simple word tokenization
        toks = re.findall(r'\b[a-zA-Z0-9_]+\b', str(t).lower())
        words.extend(toks)
    return Counter(words).most_common(top_n)

# Check France in test
s1_test = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', nrows=50000)
fr_s1 = s1_test[s1_test['country'] == 'France']
print("France S1 sample names:", fr_s1['business_name'].head(10).tolist())
print("France S1 sample addrs:", fr_s1['business_address'].head(10).tolist())
print("France S1 top name tokens:", get_tokens(fr_s1['business_name'], 30))
print("France S1 top addr tokens:", get_tokens(fr_s1['business_address'], 30))

in_s1 = s1_test[s1_test['country'] == 'India']
print("\nIndia S1 sample names:", in_s1['business_name'].head(5).tolist())
print("India S1 sample addrs:", in_s1['business_address'].head(5).tolist())
print("India S1 top name tokens:", get_tokens(in_s1['business_name'], 25))
print("India S1 top addr tokens:", get_tokens(in_s1['business_address'], 25))

us_s1 = s1_test[s1_test['country'] == 'US']
print("\nUS S1 sample names:", us_s1['business_name'].head(5).tolist())
print("US S1 sample addrs:", us_s1['business_address'].head(5).tolist())
print("US S1 top name tokens:", get_tokens(us_s1['business_name'], 25))
print("US S1 top addr tokens:", get_tokens(us_s1['business_address'], 25))
