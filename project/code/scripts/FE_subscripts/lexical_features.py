from nltk.util import ngrams
from itertools import combinations
from nltk.corpus import brown, stopwords, wordnet as wn, wordnet_ic

from jellyfish import jaro_similarity, jaro_winkler_similarity

from .feature_utils import (
    jaccard_similarity,
    containment_similarity,
    information_content,
    harmonic_mean
)

# --------------------------- #
# LEXICAL FEATURE EXTRACTION  #
# --------------------------- #

def lexical_features_extractor(s1_proc, s2_proc):
    """
    Lexical Features include:
    - String-based measures (from UKP):
        - Longest Common Substring
        - Longest Common Subsequence
        - Greedy String Tiling
        - Character n-grams
        - Word n-grams (with specific configurations)
        - Levenshtein Distance
        - Jaro, Jaro-Winkler, Monge-Elkan distances
    - N-gram overlap features (from TakeLab):
        - N-gram overlap (unigrams, bigrams, trigrams)
        - Content word n-gram overlap
        - Skip n-gram overlaps (skip bigrams and trigrams)
    - WordNet-augmented word overlap (from TakeLab)
    - Weighted word overlap (from TakeLab)
    """
    features = {}

    # Tokens
    tokens1 = s1_proc['tokens_lower']
    tokens2 = s2_proc['tokens_lower']
    s1_text = s1_proc['text_no_space']
    s2_text = s2_proc['text_no_space']

    # --- UKP Lexical Features ---

    # Longest Common Substring (UKP) at character level without spaces
    lcs_length = longest_common_substring_chars(s1_text, s2_text)
    max_length = max(len(s1_text), len(s2_text))
    lcs_norm = lcs_length / max_length if max_length > 0 else 0.0
    features['longest_common_substring_norm'] = lcs_norm

    # Longest Common Subsequence (UKP) with two normalizations
    lcsq_length = longest_common_subsequence_length(tokens1, tokens2)
    lcsq_norm1 = lcsq_length / len(tokens1) if len(tokens1) > 0 else 0.0
    lcsq_norm2 = lcsq_length / len(tokens2) if len(tokens2) > 0 else 0.0
    lcsq_norm_avg = (lcsq_norm1 + lcsq_norm2) / 2
    features['longest_common_subsequence_norm'] = lcsq_norm_avg

    # Greedy String Tiling (UKP)
    gst_score = greedy_string_tiling(tokens1, tokens2, min_match_length=3)
    features['greedy_string_tiling'] = gst_score

    # Character n-grams (n=2 to 15) (UKP)
    for n in range(2, 16):
        cngrams1 = set(ngrams(s1_text, n))
        cngrams2 = set(ngrams(s2_text, n))
        jaccard = jaccard_similarity(cngrams1, cngrams2)
        features[f'char_{n}gram_jaccard'] = jaccard

    # Word n-grams (with specific configurations) (UKP)
    # Word 1- and 2-grams (Containment, without stopwords)
    for n in [1, 2]:
        ngrams1 = set(ngrams(s1_proc['tokens_no_stop'], n))
        ngrams2 = set(ngrams(s2_proc['tokens_no_stop'], n))
        containment = containment_similarity(ngrams1, ngrams2)
        features[f'word_{n}gram_containment_no_stop'] = containment

    # Word 1-, 3-, and 4-grams (Jaccard)
    for n in [1, 3, 4]:
        ngrams1 = set(ngrams(s1_proc['tokens_lower'], n))
        ngrams2 = set(ngrams(s2_proc['tokens_lower'], n))
        jaccard = jaccard_similarity(ngrams1, ngrams2)
        features[f'word_{n}gram_jaccard'] = jaccard

    # Word 2- and 4-grams (Jaccard, without stopwords)
    for n in [2, 4]:
        ngrams1 = set(ngrams(s1_proc['tokens_no_stop'], n))
        ngrams2 = set(ngrams(s2_proc['tokens_no_stop'], n))
        jaccard = jaccard_similarity(ngrams1, ngrams2)
        features[f'word_{n}gram_jaccard_no_stop'] = jaccard

    # Levenshtein Distance (UKP)
    lev_distance = levenshtein_distance(s1_text, s2_text)
    max_len = max(len(s1_text), len(s2_text))
    lev_norm = 1 - (lev_distance / max_len) if max_len > 0 else 0.0
    features['levenshtein_normalized'] = lev_norm

    # Jaro Similarity (UKP)
    jaro_sim = jaro_similarity(s1_text, s2_text)
    features['jaro_similarity'] = jaro_sim

    # Jaro-Winkler Similarity (UKP)
    jaro_winkler_sim = jaro_winkler_similarity(s1_text, s2_text)
    features['jaro_winkler_similarity'] = jaro_winkler_sim

    # Monge-Elkan Similarity (UKP)
    monge_elkan_sim = monge_elkan_similarity(s1_proc['tokens'], s2_proc['tokens'])
    features['monge_elkan_similarity'] = monge_elkan_sim


    # --- TakeLab Lexical Features ---

    # N-gram overlap (unigrams, bigrams, trigrams) (TakeLab)
    for n in [1, 2, 3]:
        ngrams1 = set(ngrams(s1_proc['tokens_lower_nopunct'], n))
        ngrams2 = set(ngrams(s2_proc['tokens_lower_nopunct'], n))
        overlap = takelab_ngram_overlap(ngrams1, ngrams2)
        features[f'ngram_overlap_{n}'] = overlap

    # Content word n-gram overlap (TakeLab)
    for n in [1, 2, 3]:
        ngrams1 = set(ngrams(s1_proc['content_tokens'], n))
        ngrams2 = set(ngrams(s2_proc['content_tokens'], n))
        overlap = takelab_ngram_overlap(ngrams1, ngrams2)
        features[f'content_ngram_overlap_{n}'] = overlap

    # Skip n-gram overlaps (skip bigrams and trigrams) (TakeLab)
    for n in [2, 3]:
        skip_ngrams1 = set(skip_ngrams(s1_proc['tokens_lower_nopunct'], n))
        skip_ngrams2 = set(skip_ngrams(s2_proc['tokens_lower_nopunct'], n))
        overlap = takelab_ngram_overlap(skip_ngrams1, skip_ngrams2)
        features[f'skip_ngram_overlap_{n}'] = overlap

    # WordNet-augmented word overlap (TakeLab)
    S1 = s1_proc['lemmas_no_stop']
    S2 = s2_proc['lemmas_no_stop']
    pwn_s1s2 = P_WN(S1, S2)
    pwn_s2s1 = P_WN(S2, S1)
    wn_aug_overlap = harmonic_mean(pwn_s1s2, pwn_s2s1)
    features['wordnet_augmented_overlap'] = wn_aug_overlap

    # Weighted word overlap (TakeLab)
    weighted_overlap = weighted_word_overlap(S1, S2)
    features['weighted_word_overlap'] = weighted_overlap

    features = {f'lex_{key}': value for key, value in features.items()}

    return features

# --------------------------- #
#           FUNCTIONS         #
# --------------------------- #

def longest_common_substring_chars(s1_text, s2_text):
    """
    Compute the length of the longest common substring between two strings (character-based, without spaces).

    Args:
        s1_text (str): Text from s1 with no spaces.
        s2_text (str): Text from s2 with no spaces.

    Returns:
        int: Length of the longest common substring.
    """
    s1 = s1_text
    s2 = s2_text
    m = len(s1)
    n = len(s2)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                c = table[i][j] + 1
                table[i+1][j+1] = c
                if c > longest:
                    longest = c
    return longest

def longest_common_subsequence_length(s1_tokens, s2_tokens):
    """
    Compute the length of the longest common subsequence between two lists of tokens.

    Args:
        s1_tokens (list): List of tokens from s1.
        s2_tokens (list): List of tokens from s2.

    Returns:
        int: Length of the longest common subsequence.
    """
    m = len(s1_tokens)
    n = len(s2_tokens)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if s1_tokens[i] == s2_tokens[j]:
                L[i + 1][j + 1] = L[i][j] + 1
            else:
                L[i + 1][j + 1] = max(L[i + 1][j], L[i][j + 1])
    return L[m][n]

def greedy_string_tiling(s1_tokens, s2_tokens, min_match_length=3):
    """
    Implement simplified Greedy String Tiling algorithm.

    Args:
        s1_tokens (list): List of tokens from s1.
        s2_tokens (list): List of tokens from s2.
        min_match_length (int): Minimum match length.

    Returns:
        float: Similarity score.
    """
    tiles = []
    matches = []

    len1 = len(s1_tokens)
    len2 = len(s2_tokens)
    used1 = [False] * len1
    used2 = [False] * len2

    max_match = min_match_length
    while max_match >= min_match_length:
        max_match = min_match_length - 1
        new_matches = []

        for i in range(len1):
            for j in range(len2):
                k = 0
                while i + k < len1 and j + k < len2:
                    if s1_tokens[i + k] != s2_tokens[j + k] or used1[i + k] or used2[j + k]:
                        break
                    k += 1
                if k > max_match:
                    max_match = k
                    new_matches = [(i, j, k)]

                elif k == max_match:
                    new_matches.append((i, j, k))

        for match in new_matches:
            i, j, k = match
            for m in range(k):
                used1[i + m] = True
                used2[j + m] = True
            tiles.append(k)

    similarity = (2 * sum(tiles)) / (len1 + len2) if (len1 + len2) > 0 else 0.0
    return similarity

def levenshtein_distance(s1, s2):
    """
    Compute the Levenshtein distance between two strings.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        int: Levenshtein distance.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i +1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j +1] +1
            deletions = current_row[j] +1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def monge_elkan_similarity(list1, list2):
    """
    Compute the Monge-Elkan similarity between two lists of strings.

    Args:
        list1 (list): List of strings from s1.
        list2 (list): List of strings from s2.

    Returns:
        float: Monge-Elkan similarity.
    """
    sim_sum = 0.0
    for word1 in list1:
        max_sim = 0.0
        for word2 in list2:
            sim = jaro_winkler_similarity(word1, word2)
            if sim > max_sim:
                max_sim = sim
        sim_sum += max_sim
    avg_sim = sim_sum / len(list1) if list1 else 0.0
    return avg_sim

def takelab_ngram_overlap(s1_ngrams, s2_ngrams):
    """
    Compute n-gram overlap measure as defined in TakeLab.

    Args:
        s1_ngrams (set): Set of n-grams from s1.
        s2_ngrams (set): Set of n-grams from s2.

    Returns:
        float: N-gram overlap measure.
    """
    overlap = s1_ngrams.intersection(s2_ngrams)
    if not overlap:
        return 0.0
    coverage1 = len(overlap) / len(s1_ngrams) if s1_ngrams else 0.0
    coverage2 = len(overlap) / len(s2_ngrams) if s2_ngrams else 0.0
    return harmonic_mean(coverage1, coverage2)


def skip_ngrams(tokens, n):
    """
    Generate skip n-grams from the token list.

    Args:
        tokens (list): List of tokens.
        n (int): n-gram size.

    Returns:
        set: Set of skip n-grams.
    """

    indices = range(len(tokens))
    return set(tuple(tokens[i] for i in combo) for combo in combinations(indices, n))

def P_WN(S1, S2):
    """
    Compute P_WN(S1, S2) as defined in TakeLab paper.

    Args:
        S1 (list): List of words from sentence 1.
        S2 (list): List of words from sentence 2.

    Returns:
        float: P_WN(S1, S2) value.
    """
    score_total = 0.0
    for w1 in S1:
        if w1 in S2:
            score_total += 1.0
        else:
            max_sim = 0.0
            for w2 in S2:
                sim = wordnet_path_similarity(w1, w2) or 0.0
                if sim > max_sim:
                    max_sim = sim
            score_total += max_sim
    if len(S1) == 0:
        return 0.0
    return score_total / len(S1)

def wordnet_path_similarity(w1, w2):
    """
    Compute the maximum WordNet path similarity between two words.

    Args:
        w1 (str): First word.
        w2 (str): Second word.

    Returns:
        float: Maximum path similarity between two words.
    """
    synsets1 = wn.synsets(w1)
    synsets2 = wn.synsets(w2)
    max_sim = 0.0
    for syn1 in synsets1:
        for syn2 in synsets2:
            if syn1.pos() == syn2.pos():
                sim = syn1.path_similarity(syn2)
                if sim is not None and sim > max_sim:
                    max_sim = sim
    return max_sim


def weighted_word_overlap(S1, S2):
    """
    Compute the weighted word overlap between two sentences.

    Args:
        S1 (list): List of words from sentence 1.
        S2 (list): List of words from sentence 2.

    Returns:
        float: Weighted word overlap.
    """
    ic_intersect = sum(information_content(w) for w in set(S1).intersection(set(S2)))
    ic_total = sum(information_content(w) for w in set(S1 + S2))
    return (2 * ic_intersect / ic_total) if ic_total > 0 else 0.0