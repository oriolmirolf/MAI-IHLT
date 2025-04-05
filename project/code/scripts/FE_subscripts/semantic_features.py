import nltk

from .feature_utils import (
    wordnet_similarity,
    information_content,
    harmonic_mean,
    
)

import spacy
nlp = spacy.load('en_core_web_sm')

# ---------------------------- #
# SEMANTIC FEATURE EXTRACTION  #
# ---------------------------- #

def semantic_features_extractor(s1_proc, s2_proc, esa_model, lexsub_model, smt_model, dist_thesaurus):
    """
    Semantic Features include:
    - Pairwise Word Similarity (from UKP)
    - Explicit Semantic Analysis (from UKP)
    - Lexical Substitution System (from UKP)
    - Statistical Machine Translation Back-Translation Similarity (from UKP)
    - Distributional Thesaurus (from UKP)
    - Greedy Lemma Aligning Overlap (from TakeLab)
    - Numbers Overlap (from TakeLab)
    - Named Entity Features (from TakeLab)
    - Normalized Differences (from TakeLab)
    """
    features = {}

    # --- UKP Semantic Features ---

    # Pairwise Word Similarity (Resnik measure) (UKP)
    s1_words = s1_proc['lemmas_no_stop']
    s2_words = s2_proc['lemmas_no_stop']
    avg_word_sim = pairwise_word_similarity_resnik(s1_words, s2_words)
    features['avg_word_similarity'] = avg_word_sim

    # Explicit Semantic Analysis (UKP)
    esa_sim = esa_model.compute_similarity(s1_proc['text'], s2_proc['text'])
    features['esa_similarity'] = esa_sim

    # Lexical Substitution System (UKP)
    lexsub_s1 = lexsub_model.get_substituted_tokens(s1_proc)
    lexsub_s2 = lexsub_model.get_substituted_tokens(s2_proc)
    extended_s1 = s1_proc['tokens'] + lexsub_s1
    extended_s2 = s2_proc['tokens'] + lexsub_s2
    avg_word_sim_lexsub = pairwise_word_similarity_resnik(extended_s1, extended_s2)
    features['avg_word_similarity_lexsub'] = avg_word_sim_lexsub

    # Statistical Machine Translation Back-Translation Similarity (UKP)
    smt_s1 = smt_model.backtranslate(s1_proc['text'])
    smt_s2 = smt_model.backtranslate(s2_proc['text'])
    combined_s1 = s1_proc['tokens'] + nltk.word_tokenize(smt_s1)
    combined_s2 = s2_proc['tokens'] + nltk.word_tokenize(smt_s2)
    avg_word_sim_smt = pairwise_word_similarity_resnik(combined_s1, combined_s2)
    features['avg_word_similarity_smt'] = avg_word_sim_smt

    # Distributional Thesaurus (UKP)
    dt_similarity = dist_thesaurus.dt_similarity(s1_proc['tokens'], s2_proc['tokens'])
    features['dt_similarity_cardinal_numbers'] = dt_similarity

    # --- TakeLab Semantic Features ---

    # Greedy Lemma Aligning Overlap (TakeLab)
    aligned_pairs = greedy_lemma_alignment(s1_proc['lemmas_no_stop'], s2_proc['lemmas_no_stop'])
    sim_sum = 0.0
    for (w1, w2) in aligned_pairs:
        sim = wordnet_similarity(w1, w2)
        ic = max(information_content(w1), information_content(w2))
        if sim:
            sim_sum += ic * sim
    n = max(len(s1_proc['lemmas_no_stop']), len(s2_proc['lemmas_no_stop']))
    glao_score = sim_sum / n if n > 0 else 0.0
    features['greedy_lemma_align_overlap'] = glao_score

    # Numbers Overlap (TakeLab)
    nums1 = s1_proc['numbers']
    nums2 = s2_proc['numbers']
    num_overlap = numbers_overlap(nums1, nums2)
    features['numbers_overlap'] = num_overlap

    # Named Entity Features (TakeLab)
    ne_features = named_entity_features(s1_proc['text'], s2_proc['text'])
    features.update(ne_features)

    # Normalized Differences (TakeLab)
    norm_diff_features = normalized_differences(s1_proc, s2_proc)
    features.update(norm_diff_features)

    features = {f'sem_{key}': value for key, value in features.items()}

    return features

# --------------------------- #
#           FUNCTIONS         #
# --------------------------- #

def pairwise_word_similarity_resnik(s1_words, s2_words):
    """
    Compute the average pairwise word similarity between two sentences using Resnik's measure.

    Args:
        s1_words (list): List of words from sentence 1.
        s2_words (list): List of words from sentence 2.

    Returns:
        float: Average pairwise word similarity.
    """
    sim_sum = 0.0
    count = 0
    for w1 in s1_words:
        max_sim = 0.0
        for w2 in s2_words:
            sim = wordnet_similarity(w1, w2)
            if sim:
                if sim > max_sim:
                    max_sim = sim
        sim_sum += max_sim
        count += 1
    avg_sim = sim_sum / count if count > 0 else 0.0
    return avg_sim

def greedy_lemma_alignment(s1_words, s2_words):
    """
    Perform greedy lemma alignment between two lists of words.

    Args:
        s1_words (list): List of words from sentence 1.
        s2_words (list): List of words from sentence 2.

    Returns:
        list of tuples: List of aligned word pairs.
    """
    unaligned_s1 = s1_words.copy()
    unaligned_s2 = s2_words.copy()
    aligned_pairs = []
    while unaligned_s1 and unaligned_s2:
        max_sim = 0.0
        best_pair = None
        for w1 in unaligned_s1:
            for w2 in unaligned_s2:
                sim = wordnet_similarity(w1, w2)
                if sim and sim > max_sim:
                    max_sim = sim
                    best_pair = (w1, w2)
        if best_pair:
            aligned_pairs.append(best_pair)
            unaligned_s1.remove(best_pair[0])
            unaligned_s2.remove(best_pair[1])
        else:
            break  # No more similarity can be found
    return aligned_pairs

def numbers_overlap(nums1, nums2):
    """
    Compute overlap of numbers between two sentences.

    Args:
        nums1 (list): Numbers in sentence 1.
        nums2 (list): Numbers in sentence 2.

    Returns:
        float: Numbers overlap measure.
    """
    nums1 = set(nums1)
    nums2 = set(nums2)
    overlap = nums1.intersection(nums2)
    if not overlap:
        return 0.0
    coverage1 = len(overlap) / len(nums1) if nums1 else 0.0
    coverage2 = len(overlap) / len(nums2) if nums2 else 0.0
    return harmonic_mean(coverage1, coverage2)

def named_entity_features(s1, s2):
    """
    Compute named entity features.

    Args:
        s1 (str): Sentence 1.
        s2 (str): Sentence 2.

    Returns:
        dict: Dictionary of named entity features.
    """
    features = {}
    doc1 = nlp(s1)
    doc2 = nlp(s2)

    ents1 = set([(ent.text, ent.label_) for ent in doc1.ents])
    ents2 = set([(ent.text, ent.label_) for ent in doc2.ents])

    # overall NE overlap
    overlap = ents1.intersection(ents2)
    ne_overlap = len(overlap) / ((len(ents1) + len(ents2)) / 2) if (len(ents1) + len(ents2)) > 0 else 0.0
    features['named_entity_overlap'] = ne_overlap

    # NE class specific overlap
    ne_types = set([ent.label_ for ent in doc1.ents] + [ent.label_ for ent in doc2.ents])
    for ne_type in ne_types:
        ents1_type = set([ent.text for ent in doc1.ents if ent.label_ == ne_type])
        ents2_type = set([ent.text for ent in doc2.ents if ent.label_ == ne_type])
        overlap = ents1_type.intersection(ents2_type)
        if ents1_type or ents2_type:
            coverage1 = len(overlap) / len(ents1_type) if ents1_type else 0.0
            coverage2 = len(overlap) / len(ents2_type) if ents2_type else 0.0
            ne_type_overlap = harmonic_mean(coverage1, coverage2)
            features[f'named_entity_overlap_{ne_type}'] = ne_type_overlap

    return features

def normalized_differences(s1_proc, s2_proc):
    """
    Compute normalized differences features.

    Args:
        s1_proc (dict): Preprocessed sentence 1.
        s2_proc (dict): Preprocessed sentence 2.

    Returns:
        dict: Dictionary of normalized difference features.
    """
    features = {}

    # Sentence Length Difference (TakeLab)
    len1 = len(s1_proc['tokens'])
    len2 = len(s2_proc['tokens'])
    length_diff = abs(len1 - len2) / ((len1 + len2) / 2) if (len1 + len2) > 0 else 0.0
    features['length_difference'] = length_diff

    # Aggregate Word Information Content Difference (TakeLab)
    ic_s1 = sum(information_content(w) for w in s1_proc['lemmas_no_stop'])
    ic_s2 = sum(information_content(w) for w in s2_proc['lemmas_no_stop'])
    ic_diff = abs(ic_s1 - ic_s2) / ((ic_s1 + ic_s2) / 2) if (ic_s1 + ic_s2) > 0 else 0.0
    features['information_content_difference'] = ic_diff

    return features