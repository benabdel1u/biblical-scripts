# sefaria_merge_v14.py

import pandas as pd
import requests
import re
import time
import unicodedata
from tqdm import tqdm
import nltk
from collections import defaultdict, Counter



# -------------------------
# Configuration
# -------------------------
HEBREW_FILE = 'list-verbs-MT-July-2025.xlsx'
SYRIAC_FILE = 'list_verbs_Peshitta_July_2025_v3.xlsx'
OUTPUT_FILE = 'merged_verbs_results_v14.csv'
ERROR_REPORT_FILE = 'error_report_v14.csv'

# Which Syriac columns to export back to the merged output
SYRIAC_EXPORT_COLS = ['Verb_tense', 'VT', 'PS', 'VO']


# -------------------------
# Caches and dictionaries
# -------------------------
sefaria_cache = {}   # Map hebrew_root (string of Hebrew chars) -> list of candidate Syriac canonical roots (or [] if none)
wordnet_cache = {}   # Cache gloss -> synonym set (lowercased strings)
canonical_cache = {} # Cache translit root -> frozenset of canonical variants

# Expanded transliteration maps
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
    # Some composed sequences seen in sources:
    'ܟܹ': 'K', 'ܡܹ': 'M', 'ܢܹ': 'N', 'ܦܹ': 'P', 'ܨܹ': 'Y'
}

# Root Synonym map
ROOT_ALIASES = {
    'HRH': {'BVN'},
    'NTN': {'CLM'},
    'NC>': {'NC>', 'NCJ'},
    'RDH': {'CLV'},
    'MCL': {'CLV'},
    'CLV': {'CLV', 'MCL', 'RDH'},
    'BW>': {'>TJ'},
}

# Curated synonym maps
CURATED_SYNONYMS = {
    'be good': {'good', 'fair', 'pleasing'},
    'grow green': {'sprout', 'green', 'grow', 'bud'},
    'sprout': {'sprout', 'grow', 'bud', 'shoot'},
    'beguile': {'deceive', 'mislead', 'seduce', 'beguile'},
    'rule': {'rule', 'govern', 'reign', 'dominate'},
    'shine': {'shine', 'give light', 'illuminate', 'light'},
    'be holy': {'sanctify', 'holy', 'consecrate'},
    'be many': {'increase', 'multiply', 'be many', 'be numerous'},
    'bear fruit': {'bear fruit', 'be fruitful', 'be fertile'},
    'separate': {'separate', 'divide', 'split'},
    'go out': {'go out', 'depart', 'leave'},
    'be ashamed': {'be ashamed', 'shame'},
    'rest': {'rest', 'cease'},
    'work': {'work', 'serve', 'labor'},
    'bless': {'bless'},
    'create': {'create'},
    'call': {'call', 'read'},
    'see': {'see', 'look'},
    'make': {'make', 'do'},
    'eat': {'eat'},
    'die': {'die'},
    'know': {'know'},
    'take': {'take', 'seize'},
    'give': {'give'},
    'come': {'come', 'enter'},
    'send': {'send'},
    'keep': {'keep', 'guard'},
    'walk': {'walk'},
    'hide': {'hide'},
    'open': {'open'},
    'sleep': {'sleep'},
    'ascend': {'ascend', 'go up'},
    'descend': {'descend', 'go down'},
    'return': {'return', 'turn'},
    'fill': {'fill', 'be full'},
    'sow': {'sow'},
    'walk': {'walk', 'go'},
    'lift':{'lift', 'leave', 'be high'},
    'speak': {'speak', 'say'},
    'be able': {'be able', 'find'},
    'destroy':{'destroy', 'corrupt'},
}

# Stopwords
STOPWORDS = {
    'be', 'do', 'see', 'make', 'know', 'take', 'come', 'give',
    'to', 'a', 'the', 'of', 'from', 'that', 'and', 'or', 'will', 'is', 'are',
    'by', 'with', 'for', 'on', 'in', 'at', 'as'
}


# -------------------------
# Setup functions
# -------------------------
def setup_nltk():
    """Ensure WordNet is available."""
    try:
        nltk.data.find('corpora/wordnet.zip')
        print("NLTK WordNet is already installed.")
    except LookupError:
        print("WordNet resource not found. Downloading...")
        try:
            nltk.download('wordnet')
            print("Download of WordNet complete.")
        except Exception as e:
            print(f"Could not download WordNet: {e}. Proceeding with curated synonyms only.")


# -------------------------
# Text normalization and transliteration
# -------------------------
def remove_hebrew_diacritics(text):
    """Remove Hebrew niqqud/taamim marks."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'[\u0591-\u05C7]', '', text)


def transliterate(text, translit_map):
    """Generic transliteration by map table."""
    if not isinstance(text, str):
        return ""
    return "".join(translit_map.get(char, '') for char in text)


def transliterate_hebrew(hebrew_root):
    """Transliterate a Hebrew root to the unified Latin code."""
    cleaned_root = remove_hebrew_diacritics(hebrew_root)
    return transliterate(cleaned_root, HEBREW_TRANSLIT_MAP)


def normalize_syriac_unicode(syriac_text):
    """Normalize and strip Syriac diacritics."""
    if not isinstance(syriac_text, str):
        return ""
    text = unicodedata.normalize('NFC', syriac_text)
    text = re.sub(r'[\u0730-\u074A]', '', text)
    return text


def transliterate_syriac(syriac_text):
    """Normalize and transliterate Syriac."""
    normalized_text = normalize_syriac_unicode(syriac_text)
    return transliterate(normalized_text, SYRIAC_TRANSLIT_MAP)


# -------------------------
# Canonicalization
# -------------------------
def expand_aliases_for_root(root):
    """Fetch known alias set from ROOT_ALIASES."""
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
    """Handle ">" <-> "H" interchange."""
    variants = {form}
    if form.startswith('>'):
        variants.add('H' + form[1:])
    elif form.startswith('H'):
        variants.add('>' + form[1:])
    return variants


def normalize_s_sh_c_collisions(form):
    """Handle S <-> C interchange."""
    variants = {form}
    swapped = form.translate(str.maketrans({'S': 'C', 'C': 'S'}))
    variants.add(swapped)
    return variants


def normalize_final_matres(form):
    """Normalize final weak letters."""
    variants = {form}
    if form.endswith('>') or form.endswith('H'):
        variants.add(form[:-1] + 'J')
    if form.endswith('W') and len(form) > 1:
        variants.add(form[:-1])
    return variants


def create_canonical_variants(translit_root):
    """Create a set of canonical variants for robust matching."""
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
    """Fetch Syriac cognates from Sefaria API, collecting all matches."""
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
# Semantics and synonyms
# -------------------------
def get_synonym_set(gloss):
    """Generate a robust synonym set from a gloss string."""
    if not isinstance(gloss, str) or not gloss.strip():
        return set()
    if gloss in wordnet_cache:
        return wordnet_cache[gloss]

    from nltk.corpus import wordnet

    cleaned = gloss.replace('"', '').strip().lower()
    parts = [p.strip() for p in re.split(r'[;,/–—-]', cleaned) if p.strip()]
    synonyms = set()

    for part in parts:
        if part in CURATED_SYNONYMS:
            synonyms |= {w.lower() for w in CURATED_SYNONYMS[part]}

    for part in parts:
        tokens = re.findall(r"[a-z']+", part)
        for tok in tokens:
            if tok and tok not in STOPWORDS:
                synonyms.add(tok)
    try:
        for tok in list(synonyms):
            for syn in wordnet.synsets(tok):
                for lemma in syn.lemmas():
                    w = lemma.name().replace('_', ' ').lower()
                    if w and w not in STOPWORDS:
                        synonyms.add(w)
    except Exception:
        pass

    wordnet_cache[gloss] = synonyms
    return synonyms


# -------------------------
# Helpers for scoring and tie-breaking
# -------------------------
def share_canonical_variant(heb_variants, syr_variants):
    """Check for intersection between two sets of canonical variants."""
    return len(heb_variants.intersection(syr_variants)) > 0


def vo_vt_ps_similarity(heb_row, syr_row):
    """Compute a morphology similarity bonus score for tie-breaking."""
    score = 0
    for col in ['VO', 'VT', 'PS']:
        h = str(heb_row.get(col.lower(), '') or '').strip().lower()
        s = str(syr_row.get(col.lower(), '') or '').strip().lower()
        if h and s and h == s:
            score += 1
    return score


def nearest_syriac_by_canonical(heb_variants, available_syr_df):
    """Find Syriac candidates sharing a canonical variant."""
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
    setup_nltk()

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
    not_found_records = []
    
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
            rejection_reasons = []

            heb_canons = heb_row.get('canonical_variants', frozenset())
            heb_gloss = heb_row.get('gloss', '')
            heb_synonyms = get_synonym_set(heb_gloss)

            if syr_candidates_df.empty:
                rejection_reasons.append("No Syriac candidates in this verse.")
            else:
                available_syr_df = syr_candidates_df[~syr_candidates_df.index.isin(used_syr_indices)]
                if available_syr_df.empty:
                    rejection_reasons.append("All Syriac candidates already used in this verse.")
                else:
                    # STRATEGY 1: Canonical Root Match
                    canon_matches = nearest_syriac_by_canonical(heb_canons, available_syr_df)
                    if canon_matches:
                        idx, syriac_match_row = canon_matches[0]
                        match_method, match_found = 'Canonical Match', True
                    else:
                        rejection_reasons.append("No canonical variant intersection found.")

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
                            else:
                                rejection_reasons.append("Sefaria cognates found, but none in this verse.")
                        else:
                            rejection_reasons.append("Sefaria returned no cognates.")

                    # STRATEGY 3: Semantic Match
                    if not match_found and not available_syr_df.empty:
                        best_match, best_score, best_tiebreak = None, -1, (-1, -1)
                        for idx3, row3 in available_syr_df.iterrows():
                            overlap = len(heb_synonyms.intersection(get_synonym_set(row3.get('gloss', ''))))
                            if overlap > 0:
                                canonical_bonus = 1 if share_canonical_variant(heb_canons, row3.get('canonical_variants', frozenset())) else 0
                                morph_bonus = vo_vt_ps_similarity(heb_row, row3)
                                tie_tuple = (canonical_bonus, morph_bonus)
                                if overlap > best_score or (overlap == best_score and tie_tuple > best_tiebreak):
                                    best_score, best_tiebreak, best_match = overlap, tie_tuple, (idx3, row3)
                        if best_match:
                            idx, syriac_match_row = best_match
                            match_method = f"Semantic Match ({best_score} common)"
                            match_found = True
                        else:
                            rejection_reasons.append("Semantic overlap yielded zero intersections.")

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

            if not match_found:
                not_found_records.append({
                    'lexeme-v': heb_row['lexeme-v'], 'gloss': heb_gloss, 'book': heb_row['book'],
                    'chapter': heb_row['chapter'], 'verse': heb_row['verse'],
                    'translit_root': heb_row.get('translit_root', '')
                })

    df_final = pd.DataFrame(results)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n--- SUCCESS ---\nMerged file saved as: {OUTPUT_FILE}")

    if not_found_records:
        nf_df = pd.DataFrame(not_found_records)
        counts = nf_df.groupby('lexeme-v').size().sort_values(ascending=False)
        top_20_lexemes = counts.head(20).index.tolist()
        report_rows = []
        df_syr_big = df_syr.copy()

        for lex in top_20_lexemes:
            sub_df = nf_df[nf_df['lexeme-v'] == lex]
            verses = list(dict.fromkeys(zip(sub_df['book'], sub_df['chapter'], sub_df['verse'])))
            for (book, chap, vrs) in verses:
                sub_row = sub_df[(sub_df['book'] == book) & (sub_df['chapter'] == chap) & (sub_df['verse'] == vrs)].iloc[0]
                heb_can = create_canonical_variants(sub_row['translit_root'])
                heb_syn = get_synonym_set(sub_row['gloss'])
                cand_syr = df_syr_big[(df_syr_big['book'] == book) & (df_syr_big['chapter'] == chap) & (df_syr_big['verse'] == vrs)]

                nearest_canon = []
                for _, row in cand_syr.iterrows():
                    if share_canonical_variant(heb_can, row.get('canonical_variants', frozenset())):
                        nearest_canon.append((row.get('lemma', ''), row.get('gloss', '')))
                
                # --- FIX v14: The block below was corrected for indentation and logic. ---
                # It finds the best semantic candidate for the error report.
                best_sem = ('', '', 0) # (lemma, gloss, overlap_score)
                for _, row in cand_syr.iterrows():
                    overlap = len(heb_syn.intersection(get_synonym_set(row.get('gloss', ''))))
                    # Compare new overlap with the best score so far (best_sem[2])
                    if overlap > best_sem[2]:
                        best_sem = (row.get('lemma', ''), row.get('gloss', ''), overlap)

                report_rows.append({
                    'lexeme-v': lex, 'not_found_count': counts[lex], 'book': book, 'chapter': chap, 'verse': vrs,
                    'heb_gloss': sub_row['gloss'], 'heb_translit_root': sub_row['translit_root'],
                    'heb_canons': ';'.join(sorted(heb_can)),
                    'nearest_syriac_by_canonical': '; '.join([f"{lem}:{gls}" for lem, gls in nearest_canon[:3]]) if nearest_canon else 'None',
                    'best_semantic_candidate': f"{best_sem[0]}:{best_sem[1]} ({best_sem[2]} overlap)" if best_sem[2] > 0 else 'None',
                })

        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(ERROR_REPORT_FILE, index=False, encoding='utf-8-sig')
        print(f"Error report saved to: {ERROR_REPORT_FILE}")

        print("\nTop 20 Hebrew lexemes by Not Found count:")
        for lex in top_20_lexemes:
            print(f"  {lex}: {counts[lex]}")
    else:
        print("Great! No Not Found entries to report.")

if __name__ == "__main__":
    main()
