import unicodedata
import re

LEGAL_SUFFIX_MAP = {
    # US / Global
    'incorporated': 'inc',
    'corporation': 'corp',
    'limited liability company': 'llc',
    'l l c': 'llc',
    'llc': 'llc',
    'limited liability partnership': 'llp',
    'l l p': 'llp',
    'llp': 'llp',
    'limited': 'ltd',
    'company': 'co',
    'professional corporation': 'pc',
    'p c': 'pc',
    'professional limited liability company': 'pllc',
    'pllc': 'pllc',
    'limited partnership': 'lp',
    'l p': 'lp',
    # Indian
    'private limited': 'pvt ltd',
    'pvt ltd': 'pvt ltd',
    'pvt limited': 'pvt ltd',
    'private ltd': 'pvt ltd',
    'private': 'pvt',
    'pvt': 'pvt',
    'proprietorship': 'prop',
    'proprietor': 'prop',
    'prop': 'prop',
    'enterprises': 'ent',
    'industries': 'ind',
    # French
    'societe a responsabilite limitee': 'sarl',
    's a r l': 'sarl',
    'sarl': 'sarl',
    'societe par actions simplifiee': 'sas',
    's a s': 'sas',
    'sas': 'sas',
    'societe par actions simplifiee unipersonnelle': 'sasu',
    's a s u': 'sasu',
    'sasu': 'sasu',
    'entreprise unipersonnelle a responsabilite limitee': 'eurl',
    'e u r l': 'eurl',
    'eurl': 'eurl',
    'societe anonyme': 'sa',
    's a': 'sa',
    'sa': 'sa',
    'societe civile immobiliere': 'sci',
    's c i': 'sci',
    'sci': 'sci',
    'societe en nom collectif': 'snc',
    's n c': 'snc',
    'snc': 'snc',
    'etablissements': 'ets',
    'etablissement': 'ets',
    'ets': 'ets',
    'compagnie': 'cie',
    'cie': 'cie',
    'societe': 'soc',
    'soc': 'soc',
    'association': 'assoc',
}

ADDRESS_MAP = {
    # US / Global
    'street': 'st',
    'str': 'st',
    'road': 'rd',
    'avenue': 'ave',
    'av': 'ave',
    'boulevard': 'blvd',
    'bvd': 'blvd',
    'drive': 'dr',
    'lane': 'ln',
    'court': 'ct',
    'place': 'pl',
    'square': 'sq',
    'highway': 'hwy',
    'parkway': 'pkwy',
    'apartment': 'apt',
    'suite': 'ste',
    # Indian
    'opposite': 'opp',
    'building': 'bldg',
    'floor': 'flr',
    'fl': 'flr',
    'near': 'near',
    'nr': 'near',
    'sector': 'sec',
    'district': 'dist',
    # French
    'bd': 'blvd',
    'impasse': 'imp',
    'chemin': 'chem',
    'allee': 'all',
    'route': 'rte',
    'cours': 'crs',
}

# Compile single unified regex for legal terms (longer keys first to match multi-words before single-words)
sorted_legal_keys = sorted(LEGAL_SUFFIX_MAP.keys(), key=lambda x: -len(x))
legal_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted_legal_keys) + r')\b')

# Compile single unified regex for address terms
sorted_addr_keys = sorted(ADDRESS_MAP.keys(), key=lambda x: -len(x))
addr_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted_addr_keys) + r')\b')

NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')
WHITESPACE = re.compile(r'\s+')

def strip_accents(text: str) -> str:
    """Normalize Unicode characters and strip diacritical marks/accents."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def normalize_text(text: str, is_address: bool = False) -> str:
    """Fast, single-pass normalization of business name or address strings."""
    if not isinstance(text, str) or not text:
        return ""
    # Strip accents & lowercase
    t = strip_accents(text).lower()
    # Strip punctuation to spaces
    t = NON_ALPHANUM.sub(' ', t)
    # Canonicalize legal suffixes
    t = legal_pattern.sub(lambda m: LEGAL_SUFFIX_MAP[m.group(0)], t)
    # Canonicalize address components if specified
    if is_address:
        t = addr_pattern.sub(lambda m: ADDRESS_MAP[m.group(0)], t)
    return WHITESPACE.sub(' ', t).strip()
