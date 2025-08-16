# sefaria_merge_v16_bert.py

import pandas as pd
import requests
import re
import unicodedata
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Try to use SciPy if available; fall back to pure-Python Hungarian otherwise
try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# -------------------------
# Configuration
# -------------------------

HEBREW_FILE = 'list-verbs-MT-July-2025.xlsx'
SYRIAC_FILE = 'list_verbs_Peshitta_July_2025_v3.xlsx'
OUTPUT_FILE = 'merged_verbs_results_v16_bert.csv'

# Which Syriac columns to export back to the merged output
SYRIAC_EXPORT_COLS = ['Verb_tense', 'VT', 'PS', 'VO']

# -------------------------
# Caches and dictionaries
# -------------------------

sefaria_cache = {}  # Map hebrew_root -> list of candidate Syriac canonical roots
canonical_cache = {}  # Cache translit root -> frozenset of canonical variants
gloss_embedding_cache = {}  # Cache gloss -> vector embedding

# Transliteration and Root Alias Maps (Unchanged)

HEBREW_TRANSLIT_MAP = {
    'א': '>', 'ב': 'B', 'ג': 'G', 'ד': 'D', 'ה': 'H', 'ו': 'W', 'ז': 'Z',
    'ח': 'X', 'ט': 'V', 'י': 'J', 'כ': 'K', 'ך': 'K', 'ל': 'L', 'מ': 'M', 'ם': 'M',
    'נ': 'N', 'ן': 'N', 'ס': 'S', 'ע': '<', 'פ': 'P', 'ף': 'P', 'צ': 'Y', 'ץ': 'Y',
    'ק': 'Q', 'ר': 'R', 'שׂ': 'S', 'שׁ': 'C', 'ש': 'C', 'ת': 'T'
}

SYRIAC_TRANSLIT_MAP = {
    'ܐ': '>', 'ܒ': 'B', 'ܓ': 'G', 'ܕ': 'D', 'ܗ': 'H', 'ܘ': 'W', 'ܙ': 'Z',
    'ܚ': 'X', 'ܛ': 'V', 'ܝ': 'J', 'ܟ': 'K', 'ܠ': 'L', 'ܡ': 'M', 'ܢ': 'N',
    'ܣ': 'S', 'ܥ': '<', 'ܦ': 'P', 'ܨ': 'Y', 'ܩ': 'Q', 'ܪ': 'R', 'ܫ': 'C', 'ܬ': 'T',
    'ܟܹ': 'K', 'ܡܹ': 'M', 'ܢܹ': 'N', 'ܦܹ': 'P', 'ܨܹ': 'Y'
}

ROOT_ALIASES = {
    'HRH': {'BVN'},
    'NTN': {'CLM'},
    'RDH': {'CLV'},
    'MCL': {'CLV'},
    'CLV': {'CLV', 'MCL', 'RDH'},
    'BW>': {'>TJ'},
    'NWX': {'CBQ'},
}

# -------------------------
# Text normalization and transliteration
# -------------------------

def transliterate(text, translit_map):
    if not isinstance(text, str):
        return ""
    sorted_keys = sorted(translit_map.keys(), key=len, reverse=True)
    output = []
    i = 0
    text_len = len(text)
    while i < text_len:
        match_found = False
        for key in sorted_keys:
            if text.startswith(key, i):
                output.append(translit_map[key])
                i += len(key)
                match_found = True
                break
        if not match_found:
            i += 1
    return "".join(output)

def transliterate_hebrew(hebrew_root):
    return transliterate(hebrew_root, HEBREW_TRANSLIT_MAP)

def normalize_syriac_unicode(syriac_text):
    if not isinstance(syriac_text, str):
        return ""
    text = unicodedata.normalize('NFC', syriac_text)
    # Remove Syriac diacritics (vowel points etc.)
    text = re.sub(r'[\u0730-\u074A]', '', text)
    return text

def transliterate_syriac(syriac_text):
    normalized_text = normalize_syriac_unicode(syriac_text)
    return transliterate(normalized_text, SYRIAC_TRANSLIT_MAP)

# -------------------------
# Canonicalization
# -------------------------

def expand_aliases_for_root(root):
    aliases = set()
    if root in ROOT_ALIASES:
        aliases |= set(ROOT_ALIASES[root])
    for k, v in ROOT_ALIASES.items():
        if root in v:
            aliases.add(k)
            aliases |= set(v)
    aliases.add(root)
    return aliases

def normalize_initial_aleph_he_variation(form):
    variants = {form}
    if form.startswith('>'):
        variants.add('H' + form[1:])
    elif form.startswith('H'):
        variants.add('>' + form[1:])
    return variants

def normalize_s_sh_c_collisions(form):
    variants = {form}
    swapped = form.translate(str.maketrans({'S': 'C', 'C': 'S'}))
    variants.add(swapped)
    return variants

def normalize_final_matres(form):
    variants = {form}
    if form.endswith('>') or form.endswith('H'):
        variants.add(form[:-1] + 'J')
    if form.endswith('W') and len(form) > 1:
        variants.add(form[:-1])
    return variants

def create_canonical_variants(translit_root):
    if not isinstance(translit_root, str) or not translit_root:
        return frozenset()
    if translit_root in canonical_cache:
        return canonical_cache[translit_root]

    variants = {translit_root}
    variants |= normalize_final_matres(translit_root)

    new_variants = set()
    for v in variants:
        new_variants |= normalize_initial_aleph_he_variation(v)
    variants |= new_variants

    new_variants = set()
    for v in variants:
        new_variants |= normalize_s_sh_c_collisions(v)
    variants |= new_variants

    expanded = set()
    for v in variants:
        expanded |= expand_aliases_for_root(v)
    variants |= expanded

    canonical = frozenset(variants)
    canonical_cache[translit_root] = canonical
    return canonical

# -------------------------
# Sefaria cognates
# -------------------------

def get_sefaria_cognates(hebrew_root):
    if hebrew_root in sefaria_cache:
        return sefaria_cache[hebrew_root]
    url = f"https://www.sefaria.org/api/words/{hebrew_root}"
    headers = {'User-Agent': 'Sefaria-Merge-Script/1.0'}
    candidates = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for entry in data:
                if entry.get("parent_lexicon") == "BDB Dictionary" and entry.get("root") is True:
                    senses = entry.get('content', {}).get('senses', [])
                    for sense in senses:
                        definition_html = sense.get('definition', '')
                        matches = re.findall(r'([\u0700-\u074F]+)', definition_html)
                        for m in matches:
                            tr = transliterate_syriac(m)
                            if tr:
                                candidates.extend(list(create_canonical_variants(tr)))
    except requests.exceptions.RequestException:
        candidates = []
    candidates = sorted(set(candidates))
    sefaria_cache[hebrew_root] = candidates
    return candidates

# -------------------------
# BERT-based Semantic Similarity
# -------------------------

def get_embedding(gloss, model):
    """Generates or retrieves a cached embedding for a given gloss."""
    if not isinstance(gloss, str) or not gloss.strip():
        return None
    if gloss in gloss_embedding_cache:
        return gloss_embedding_cache[gloss]
    embedding = model.encode(gloss, convert_to_tensor=True)
    gloss_embedding_cache[gloss] = embedding
    return embedding

# -------------------------
# Scoring utilities
# -------------------------

def share_canonical_variant(heb_variants, syr_variants):
    return len(heb_variants.intersection(syr_variants)) > 0

def vo_vt_ps_similarity(heb_row, syr_row):
    score = 0
    for col in ['VO', 'VT', 'PS']:
        h = str(heb_row.get(col.lower(), '') or '').strip().lower()
        s = str(syr_row.get(col.lower(), '') or '').strip().lower()
        if h and s and h == s:
            score += 1
    return score

# --- New helpers for global optimization scoring ---

def morph_bonus_score(heb_row, syr_row, per_field=0.15):
    # Reuse VO/VT/PS similarity; scale to a small bonus
    return per_field * vo_vt_ps_similarity(heb_row, syr_row)

def canonical_overlap(heb_canons, syr_canons):
    return len(heb_canons.intersection(syr_canons)) > 0

def sefaria_overlap_for_heb(heb_row, syr_row, local_sefaria_cache):
    # Cache Sefaria cognate set per unique Hebrew root string in this verse
    heb_root_str = heb_row.get('lexeme-v', '')
    if heb_root_str not in local_sefaria_cache:
        local_sefaria_cache[heb_root_str] = set(get_sefaria_cognates(heb_root_str))
    sef_set = local_sefaria_cache[heb_root_str]
    syr_canons = syr_row.get('canonical_variants', frozenset())
    return len(sef_set.intersection(syr_canons)) > 0

def bert_similarity(heb_row, syr_row, model):
    he = get_embedding(heb_row.get('gloss', ''), model)
    se = get_embedding(syr_row.get('gloss', ''), model)
    if he is None or se is None:
        return 0.0
    try:
        return float(util.cos_sim(he, se).item())
    except Exception:
        return 0.0

def compute_pair_score(heb_row, syr_row, model, local_sefaria_cache,
                       weights=None, morph_per_field=0.15):
    """
    Returns (total_score, components_dict) combining:
    - canonical_bonus (1.0 if any canonical overlap else 0)
    - sefaria_bonus (0.6 if Sefaria overlap else 0)
    - bert_sim (cosine similarity, 0..1, scaled by bert_weight)
    - morph_bonus (VO/VT/PS matches * per-field)

    weights keys: canonical_bonus, sefaria_bonus, bert_weight
    """
    if weights is None:
        weights = dict(canonical_bonus=1.0, sefaria_bonus=0.6, bert_weight=1.0)

    heb_can = heb_row.get('canonical_variants', frozenset())
    syr_can = syr_row.get('canonical_variants', frozenset())

    can = weights['canonical_bonus'] if canonical_overlap(heb_can, syr_can) else 0.0
    sef = weights['sefaria_bonus'] if sefaria_overlap_for_heb(heb_row, syr_row, local_sefaria_cache) else 0.0
    bert_raw = bert_similarity(heb_row, syr_row, model)
    bert = weights['bert_weight'] * bert_raw
    morph = morph_per_field * vo_vt_ps_similarity(heb_row, syr_row)

    total = can + sef + bert + morph
    comps = dict(canonical_bonus=can, sefaria_bonus=sef, bert_sim=bert, morph_bonus=morph)
    return total, comps

# -------------------------
# Hungarian assignment helpers
# -------------------------

def _pad_to_square(W, pad_value=0.0):
    W = np.array(W, dtype=float)
    n, m = W.shape
    if n == m:
        return W
    if n < m:
        pad = np.full((m - n, m), pad_value, dtype=float)
        return np.vstack([W, pad])
    else:
        pad = np.full((n, n - m), pad_value, dtype=float)
        return np.hstack([W, pad])

def _hungarian_maximize_numpy(W):
    # Maximize by converting to a minimize problem: cost = max(W) - W
    W = np.array(W, dtype=float)
    if W.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    max_val = np.nanmax(W) if np.isfinite(W).any() else 0.0
    cost = max_val - W
    if '_HAVE_SCIPY' in globals() and _HAVE_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost)
        return row_ind, col_ind
    # Pure-Python Hungarian for square matrices (minimizing 'cost')
    return _hungarian_pure_python(cost)

def _hungarian_pure_python(cost):
    """
    Minimal pure-Python Hungarian (for square matrices, minimizing 'cost').
    Good enough for small per-verse matrices.
    """
    cost = np.array(cost, dtype=float)
    n, m = cost.shape
    assert n == m, "Hungarian fallback expects square matrix"
    N = n

    # Row reduction
    cost = cost - cost.min(axis=1, keepdims=True)
    # Column reduction
    cost = cost - cost.min(axis=0, keepdims=True)

    starred = np.zeros_like(cost, dtype=bool)
    primed = np.zeros_like(cost, dtype=bool)
    covered_rows = np.zeros(N, dtype=bool)
    covered_cols = np.zeros(N, dtype=bool)

    # Star one zero in each row if possible
    for i in range(N):
        for j in range(N):
            if (cost[i, j] == 0) and (not covered_rows[i]) and (not covered_cols[j]):
                starred[i, j] = True
                covered_rows[i] = True
                covered_cols[j] = True
                break
    covered_rows[:] = False
    covered_cols[:] = False

    def cover_cols_with_starred():
        covered_cols[:] = False
        for j in range(N):
            if starred[:, j].any():
                covered_cols[j] = True

    cover_cols_with_starred()

    def find_uncovered_zero():
        for i in range(N):
            if not covered_rows[i]:
                for j in range(N):
                    if (not covered_cols[j]) and (cost[i, j] == 0):
                        return i, j
        return None

    while covered_cols.sum() < N:
        pos = find_uncovered_zero()
        while pos is None:
            # Adjust matrix by smallest uncovered value
            uncovered_rows = ~covered_rows
            uncovered_cols = ~covered_cols
            min_uncovered = np.min(cost[np.ix_(uncovered_rows, uncovered_cols)])
            # Add min to covered rows
            cost[covered_rows, :] += min_uncovered
            # Subtract min from uncovered cols
            cost[:, uncovered_cols] -= min_uncovered
            pos = find_uncovered_zero()

        i, j = pos
        primed[i, j] = True
        star_col = np.where(starred[i])[0]
        if star_col.size == 0:
            # Construct alternating path
            path = [(i, j)]
            done = False
            while not done:
                r = np.where(starred[:, path[-1][1]])
                if r.size == 0:
                    done = True
                else:
                    r = int(r)
                    path.append((r, path[-1][1]))
                    c = np.where(primed[r])
                    c = int(c)
                    path.append((r, c))
            # Flip stars along the path
            for (r, c) in path:
                if starred[r, c]:
                    starred[r, c] = False
                else:
                    starred[r, c] = True
            primed[:, :] = False
            covered_rows[:] = False
            cover_cols_with_starred()
        else:
            # Cover this row and uncover the column of the starred zero
            covered_rows[i] = True
            covered_cols[int(star_col)] = False

    row_ind = np.arange(N, dtype=int)
    col_ind = np.argmax(starred, axis=1)
    return row_ind, col_ind

# -------------------------
# Main pipeline
# -------------------------

def main():
    # Load the SentenceTransformer model once
    print("Loading multilingual sentence model (LaBSE)...")
    bert_model = SentenceTransformer('sentence-transformers/LaBSE')
    print("Model loaded.")

    try:
        df_heb = pd.read_excel(HEBREW_FILE)
        df_syr = pd.read_excel(SYRIAC_FILE)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return

    df_heb.columns = df_heb.columns.str.lower().str.strip()
    df_syr.columns = df_syr.columns.str.lower().str.strip()

    print("Applying transliteration and creating canonical variants...")
    df_heb['translit_root'] = df_heb['lexeme-v'].apply(transliterate_hebrew)
    df_syr['translit_root'] = df_syr['lemma']

    df_heb['canonical_variants'] = df_heb['translit_root'].apply(create_canonical_variants)
    df_syr['canonical_variants'] = df_syr['translit_root'].apply(create_canonical_variants)

    heb_grouped = df_heb.groupby(['book', 'chapter', 'verse'])
    syr_grouped = df_syr.groupby(['book', 'chapter', 'verse'])

    results = []

    for verse_key, heb_verse_df in tqdm(heb_grouped, desc="Matching verbs"):
        try:
            syr_candidates_df = syr_grouped.get_group(verse_key).copy()
        except KeyError:
            syr_candidates_df = pd.DataFrame()

        # --------------------------------------------------------------------------
        # GLOBAL OPTIMIZATION PER VERSE (Maximum Weight Bipartite Matching)
        # --------------------------------------------------------------------------

        # If no Syriac entries for this verse, mark all as Not Found
        if syr_candidates_df.empty:
            for idx, heb_row in heb_verse_df.iterrows():
                result_row = heb_row.to_dict()
                result_row['Syriac_Lemma'] = None
                result_row['Syriac_Gloss'] = None
                for col in SYRIAC_EXPORT_COLS:
                    result_row[col] = None
                result_row['Match_Method'] = 'Not Found'
                results.append(result_row)
            continue

        # Index rows explicitly
        heb_indices = list(heb_verse_df.index)
        syr_indices = list(syr_candidates_df.index)

        H = heb_verse_df.loc[heb_indices]
        S = syr_candidates_df.loc[syr_indices]

        # Local Sefaria cache per verse for performance
        local_sefaria_cache = {}

        # Build weight matrix |H| x |S| and keep detailed component trackers
        W = np.zeros((len(H), len(S)), dtype=float)
        components = [[None for _ in range(len(S))] for __ in range(len(H))]

        for i, (_, heb_row) in enumerate(H.iterrows()):
            for j, (_, syr_row) in enumerate(S.iterrows()):
                total, comps = compute_pair_score(heb_row, syr_row, bert_model, local_sefaria_cache)
                W[i, j] = total
                components[i][j] = comps  # dict with canonical_bonus, sefaria_bonus, bert_sim, morph_bonus

        # Pad to square for fallback Hungarian
        W_square = _pad_to_square(W, pad_value=0.0)

        row_ind, col_ind = _hungarian_maximize_numpy(W_square)

        # Map assignments back; ignore matches to padded (dummy) rows/cols
        assignment_map = {}  # heb idx -> (syriac_row, total_score, comps_dict)
        for i_r, j_c in zip(row_ind, col_ind):
            if i_r >= len(H) or j_c >= len(S):
                continue
            heb_idx = heb_indices[i_r]
            syr_idx = syr_indices[j_c]
            score = float(W[i_r, j_c])
            comps = components[i_r][j_c] or dict(canonical_bonus=0.0, sefaria_bonus=0.0, bert_sim=0.0, morph_bonus=0.0)
            assignment_map[heb_idx] = (S.loc[syr_idx], score, comps)

        # Emit results in original order
        for idx, heb_row in heb_verse_df.iterrows():
            matched = assignment_map.get(idx)
            result_row = heb_row.to_dict()
            if matched is None:
                # Unassigned => Not Found
                result_row['Syriac_Lemma'] = None
                result_row['Syriac_Gloss'] = None
                for col in SYRIAC_EXPORT_COLS:
                    result_row[col] = None
                result_row['Match_Method'] = 'Not Found'
            else:
                syriac_match_row, score, comps = matched
                result_row['Syriac_Lemma'] = syriac_match_row.get('lemma')
                result_row['Syriac_Gloss'] = syriac_match_row.get('gloss')
                for col in SYRIAC_EXPORT_COLS:
                    result_row[col] = syriac_match_row.get(col.lower())
                # Compact, human-readable breakdown
                can = comps.get('canonical_bonus', 0.0)
                sef = comps.get('sefaria_bonus', 0.0)
                bert = comps.get('bert_sim', 0.0)
                morph = comps.get('morph_bonus', 0.0)
                result_row['Match_Method'] = (
                    f"Global Opt Match (Score: {score:.2f} | "
                    f"can={can:.2f}, sef={sef:.2f}, bert={bert:.2f}, morph={morph:.2f})"
                )
            results.append(result_row)

        # --------------------------------------------------------------------------
        # END GLOBAL OPTIMIZATION PER VERSE
        # --------------------------------------------------------------------------

    df_final = pd.DataFrame(results)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n--- SUCCESS ---\nMerged file saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
