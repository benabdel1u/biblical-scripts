# sefaria_merge_v15_bert.py

import pandas as pd
import requests
import re
import unicodedata
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Configuration
# -------------------------
HEBREW_FILE = 'list-verbs-MT-July-2025.xlsx'
SYRIAC_FILE = 'list_verbs_Peshitta_July_2025_v3.xlsx'
OUTPUT_FILE = 'merged_verbs_results_v15_bert.csv'

# Which Syriac columns to export back to the merged output
SYRIAC_EXPORT_COLS = ['Verb_tense', 'VT', 'PS', 'VO']

# -------------------------
# Caches and dictionaries
# -------------------------
sefaria_cache = {}  # Map hebrew_root -> list of candidate Syriac canonical roots
canonical_cache = {}  # Cache translit root -> frozenset of canonical variants
gloss_embedding_cache = {} # Cache gloss -> vector embedding

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
    if not isinstance(text, str): return ""
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
    if not isinstance(syriac_text, str): return ""
    text = unicodedata.normalize('NFC', syriac_text)
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
    if root in ROOT_ALIASES: aliases |= set(ROOT_ALIASES[root])
    for k, v in ROOT_ALIASES.items():
        if root in v:
            aliases.add(k)
            aliases |= set(v)
    aliases.add(root)
    return aliases

def normalize_initial_aleph_he_variation(form):
    variants = {form}
    if form.startswith('>'): variants.add('H' + form[1:])
    elif form.startswith('H'): variants.add('>' + form[1:])
    return variants

def normalize_s_sh_c_collisions(form):
    variants = {form}
    swapped = form.translate(str.maketrans({'S': 'C', 'C': 'S'}))
    variants.add(swapped)
    return variants

def normalize_final_matres(form):
    variants = {form}
    if form.endswith('>') or form.endswith('H'): variants.add(form[:-1] + 'J')
    if form.endswith('W') and len(form) > 1: variants.add(form[:-1])
    return variants

def create_canonical_variants(translit_root):
    if not isinstance(translit_root, str) or not translit_root: return frozenset()
    if translit_root in canonical_cache: return canonical_cache[translit_root]
    variants = {translit_root}
    variants |= normalize_final_matres(translit_root)
    new_variants = set()
    for v in variants: new_variants |= normalize_initial_aleph_he_variation(v)
    variants |= new_variants
    new_variants = set()
    for v in variants: new_variants |= normalize_s_sh_c_collisions(v)
    variants |= new_variants
    expanded = set()
    for v in variants: expanded |= expand_aliases_for_root(v)
    variants |= expanded
    canonical = frozenset(variants)
    canonical_cache[translit_root] = canonical
    return canonical

# -------------------------
# Sefaria cognates 
# -------------------------
def get_sefaria_cognates(hebrew_root):
    if hebrew_root in sefaria_cache: return sefaria_cache[hebrew_root]
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
                            if tr: candidates.extend(list(create_canonical_variants(tr)))
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
# Scoring for tie-breaking 
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

def nearest_syriac_by_canonical(heb_variants, available_syr_df):
    matches = []
    for idx, row in available_syr_df.iterrows():
        syr_var = row.get('canonical_variants', frozenset())
        if share_canonical_variant(heb_variants, syr_var):
            matches.append((idx, row))
    return matches


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

        used_syr_indices = set()

        for _, heb_row in heb_verse_df.iterrows():
            match_found = False
            match_method = 'Not Found'
            syriac_match_row = None
            
            heb_canons = heb_row.get('canonical_variants', frozenset())
            
            if not syr_candidates_df.empty:
                available_syr_df = syr_candidates_df[~syr_candidates_df.index.isin(used_syr_indices)]

                if not available_syr_df.empty:
                    # STRATEGY 1: Canonical Root Match
                    canon_matches = nearest_syriac_by_canonical(heb_canons, available_syr_df)
                    if canon_matches:
                        # If multiple canonical matches, use semantic score as a tie-breaker
                        if len(canon_matches) > 1:
                            best_semantic_score = -1
                            best_canon_match = None
                            heb_emb = get_embedding(heb_row.get('gloss', ''), bert_model)
                            if heb_emb is not None:
                                for idx_c, row_c in canon_matches:
                                    syr_emb = get_embedding(row_c.get('gloss', ''), bert_model)
                                    if syr_emb is not None:
                                        score = util.cos_sim(heb_emb, syr_emb).item()
                                        if score > best_semantic_score:
                                            best_semantic_score = score
                                            best_canon_match = (idx_c, row_c)
                            if best_canon_match:
                                idx, syriac_match_row = best_canon_match
                            else: # Fallback if embeddings fail
                                idx, syriac_match_row = canon_matches[0]
                        else: # Only one match
                            idx, syriac_match_row = canon_matches[0]
                        
                        match_method, match_found = 'Canonical Match', True

                    # STRATEGY 2: Sefaria Cognate Match
                    if not match_found:
                        sef_cands_set = set(get_sefaria_cognates(heb_row['lexeme-v']))
                        if sef_cands_set:
                            candidates = []
                            for idx2, row2 in available_syr_df.iterrows():
                                if len(sef_cands_set.intersection(row2.get('canonical_variants', frozenset()))) > 0:
                                    candidates.append((idx2, row2))
                            if candidates:
                                idx, syriac_match_row = candidates[0] 
                                match_method, match_found = 'Sefaria Canonical', True
                    
                    # STRATEGY 3: Semantic Match using BERT 
                    if not match_found:
                        heb_embedding = get_embedding(heb_row.get('gloss', ''), bert_model)

                        if heb_embedding is not None:
                            best_match, best_score, best_tiebreak = None, -1.0, -1

                            for idx3, row3 in available_syr_df.iterrows():
                                syr_embedding = get_embedding(row3.get('gloss', ''), bert_model)
                                
                                if syr_embedding is not None:
                                    # Calculate cosine similarity
                                    similarity_score = util.cos_sim(heb_embedding, syr_embedding).item()
                                    
                                    # Use morphological similarity as a tie-breaker
                                    morph_bonus = vo_vt_ps_similarity(heb_row, row3)

                                    if similarity_score > best_score or \
                                       (abs(similarity_score - best_score) < 1e-5 and morph_bonus > best_tiebreak):
                                        best_score = similarity_score
                                        best_tiebreak = morph_bonus
                                        best_match = (idx3, row3)

                            if best_match and best_score > 0.5: # Set a threshold
                                idx, syriac_match_row = best_match
                                match_method = f"Semantic Match (Score: {best_score:.2f})"
                                match_found = True

            result_row = heb_row.to_dict()
            if match_found and syriac_match_row is not None:
                result_row['Syriac_Lemma'] = syriac_match_row.get('lemma')
                result_row['Syriac_Gloss'] = syriac_match_row.get('gloss')
                for col in SYRIAC_EXPORT_COLS:
                    result_row[col] = syriac_match_row.get(col.lower())
                used_syr_indices.add(syriac_match_row.name)
            else:
                result_row['Syriac_Lemma'], result_row['Syriac_Gloss'] = None, None
                for col in SYRIAC_EXPORT_COLS:
                    result_row[col] = None
            
            result_row['Match_Method'] = match_method
            results.append(result_row)

    df_final = pd.DataFrame(results)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n--- SUCCESS ---\nMerged file saved as: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

