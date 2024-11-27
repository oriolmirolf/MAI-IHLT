# feature_extraction/syntactic_features.py

import numpy as np
import nltk
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity

from feature_extraction.feature_utils import (
    preprocess_sentence,
    get_pos_tags,
    get_dependency_relations,
    ngram_overlap_ratio,
    longest_common_subsequence,
    word_order_similarity,
    tree_edit_distance,
)


def syntactic_features(s1, s2):
    """
    Compute syntactic similarity features between two sentences.

    Derived from methods in SemEval 2012 papers.

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        dict: A dictionary of syntactic features.
    """
    features = {}
    tokens1 = preprocess_sentence(s1)
    tokens2 = preprocess_sentence(s2)

    pos_tags1_full = get_pos_tags(tokens1)
    pos_tags2_full = get_pos_tags(tokens2)

    pos_tags1 = [pos for _, pos in pos_tags1_full]
    pos_tags2 = [pos for _, pos in pos_tags2_full]

    # POS tag overlap ratio
    pos_overlap = set(pos_tags1).intersection(set(pos_tags2))
    avg_pos_length = (len(pos_tags1) + len(pos_tags2)) / 2
    pos_overlap_ratio = len(pos_overlap) / avg_pos_length if avg_pos_length != 0 else 0

    # POS tag bigram overlap
    pos_bigram_overlap = ngram_overlap_ratio(pos_tags1, pos_tags2, n=2)

    # POS tag trigram overlap
    pos_trigram_overlap = ngram_overlap_ratio(pos_tags1, pos_tags2, n=3)

    # POS tag n-gram overlap using Containment measure (n=2 to 4)
    pos_ngram_containments = {}
    for n in [2, 3, 4]:
        ngrams1 = set(ngrams(pos_tags1, n))
        ngrams2 = set(ngrams(pos_tags2, n))
        intersection = ngrams1.intersection(ngrams2)
        containment = len(intersection) / min(len(ngrams1), len(ngrams2)) \
            if min(len(ngrams1), len(ngrams2)) != 0 else 0
        pos_ngram_containments[f'syn_pos_{n}gram_containment'] = containment

    # POS tag n-gram overlap using Jaccard coefficient (n=2 to 4)
    pos_ngram_jaccards = {}
    for n in [2, 3, 4]:
        ngrams1 = set(ngrams(pos_tags1, n))
        ngrams2 = set(ngrams(pos_tags2, n))
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        jaccard = len(intersection) / len(union) if len(union) != 0 else 0
        pos_ngram_jaccards[f'syn_pos_{n}gram_jaccard'] = jaccard

    # Dependency relation overlap
    deps1 = get_dependency_relations(s1)
    deps2 = get_dependency_relations(s2)
    dep_relations1 = set([dep[0] for dep in deps1])
    dep_relations2 = set([dep[0] for dep in deps2])
    dep_overlap = dep_relations1.intersection(dep_relations2)
    dep_union = dep_relations1.union(dep_relations2)
    dep_overlap_ratio = len(dep_overlap) / len(dep_union) if len(dep_union) != 0 else 0

    # Grammatical relations proportions
    def compute_gram_rel_proportions(deps):
        total = len(deps)
        counts = {}
        for dep in deps:
            rel = dep[0]
            counts[rel] = counts.get(rel, 0) + 1
        proportions = {rel: count / total for rel, count in counts.items()}
        return proportions

    gram_rel_proportions1 = compute_gram_rel_proportions(deps1)
    gram_rel_proportions2 = compute_gram_rel_proportions(deps2)
    # Compute the cosine similarity between the grammatical relation proportions
    all_rels = set(list(gram_rel_proportions1.keys()) + list(gram_rel_proportions2.keys()))
    vector1 = [gram_rel_proportions1.get(rel, 0) for rel in all_rels]
    vector2 = [gram_rel_proportions2.get(rel, 0) for rel in all_rels]
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        gram_rel_cosine_sim = 0
    else:
        gram_rel_cosine_sim = cosine_similarity([vector1], [vector2])[0][0]

    # Word order similarity
    word_order_sim = word_order_similarity(tokens1, tokens2)

    # Longest common subsequence
    avg_length = (len(tokens1) + len(tokens2)) / 2
    lcs_length = longest_common_subsequence(tokens1, tokens2)
    lcs_norm = lcs_length / avg_length if avg_length != 0 else 0

    # Tree edit distance (simplified)
    tree_edit_dist = tree_edit_distance(s1, s2)

    # POS tag sequence similarity (normalized edit distance)
    pos_edit_distance = nltk.edit_distance(pos_tags1, pos_tags2)
    max_len = max(len(pos_tags1), len(pos_tags2))
    pos_norm_edit_distance = 1 - (pos_edit_distance / max_len) if max_len != 0 else 0

    # POS n-gram precision and recall (n=1 to 2)
    pos_ngram_precisions = []
    pos_ngram_recalls = []
    for n in range(1, 3):
        pos_ngrams1 = list(ngrams(pos_tags1, n))
        pos_ngrams2 = list(ngrams(pos_tags2, n))
        pos_ngrams1_set = set(pos_ngrams1)
        pos_ngrams2_set = set(pos_ngrams2)
        overlap = pos_ngrams1_set.intersection(pos_ngrams2_set)
        total_ngrams1 = len(pos_ngrams1)
        total_ngrams2 = len(pos_ngrams2)
        precision = len(overlap) / total_ngrams2 if total_ngrams2 != 0 else 0
        recall = len(overlap) / total_ngrams1 if total_ngrams1 != 0 else 0
        pos_ngram_precisions.append(precision)
        pos_ngram_recalls.append(recall)

    # Dependency relation precision and recall
    dep_overlap_all = set(deps1).intersection(set(deps2))
    dep_precision = len(dep_overlap_all) / len(deps2) if len(deps2) != 0 else 0
    dep_recall = len(dep_overlap_all) / len(deps1) if len(deps1) != 0 else 0

    # POS tag counts difference
    def pos_tag_counts(pos_tags):
        counts = {'N': 0, 'V': 0, 'J': 0, 'R': 0}
        for tag in pos_tags:
            if tag.startswith('N'):
                counts['N'] += 1
            elif tag.startswith('V'):
                counts['V'] += 1
            elif tag.startswith('J'):
                counts['J'] += 1
            elif tag.startswith('R'):
                counts['R'] += 1
        return counts

    counts1 = pos_tag_counts(pos_tags1)
    counts2 = pos_tag_counts(pos_tags2)

    pos_diff_features = {
        'pos_noun_diff': abs(counts1['N'] - counts2['N']),
        'pos_verb_diff': abs(counts1['V'] - counts2['V']),
        'pos_adj_diff': abs(counts1['J'] - counts2['J']),
        'pos_adv_diff': abs(counts1['R'] - counts2['R']),
    }

    # Word Pair Order
    word_pair_order_ratio = word_pair_order(tokens1, tokens2)

    # Word Pair Distance
    word_pair_distance_diff = word_pair_distance(tokens1, tokens2)

    features = {
        'syn_pos_overlap_ratio': pos_overlap_ratio,
        'syn_pos_bigram_overlap': pos_bigram_overlap,
        'syn_pos_trigram_overlap': pos_trigram_overlap,
        'syn_dep_overlap_ratio': dep_overlap_ratio,
        'syn_gram_rel_cosine_sim': gram_rel_cosine_sim,
        'syn_word_order_sim': word_order_sim,
        'syn_lcs_norm': lcs_norm,
        'syn_tree_edit_dist': tree_edit_dist,
        'syn_pos_norm_edit_distance': pos_norm_edit_distance,
        'syn_pos_ngram_precision_1': pos_ngram_precisions[0],
        'syn_pos_ngram_precision_2': pos_ngram_precisions[1],
        'syn_pos_ngram_recall_1': pos_ngram_recalls[0],
        'syn_pos_ngram_recall_2': pos_ngram_recalls[1],
        'syn_dep_precision': dep_precision,
        'syn_dep_recall': dep_recall,
        'syn_word_pair_order_ratio': word_pair_order_ratio,
        'syn_word_pair_distance_diff': word_pair_distance_diff,
    }
    features.update(pos_diff_features)
    features.update(pos_ngram_containments)
    features.update(pos_ngram_jaccards)
    return features


def word_pair_order(tokens1, tokens2):
    """
    Compute the ratio of word pairs that occur in the same order in both sentences.

    Reference: [Hatzivassiloglou et al., 1999]
    """
    pairs1 = set(zip(tokens1, tokens1[1:]))
    pairs2 = set(zip(tokens2, tokens2[1:]))
    common_pairs = pairs1.intersection(pairs2)
    total_pairs = len(pairs1.union(pairs2))
    return len(common_pairs) / total_pairs if total_pairs != 0 else 0


def word_pair_distance(tokens1, tokens2):
    """
    Compute the average absolute difference in distances between word pairs in both sentences.

    Reference: [Hatzivassiloglou et al., 1999]
    """
    word_positions1 = {word: idx for idx, word in enumerate(tokens1)}
    word_positions2 = {word: idx for idx, word in enumerate(tokens2)}
    common_words = set(word_positions1.keys()).intersection(set(word_positions2.keys()))
    if len(common_words) < 2:
        return 0
    distances1 = []
    distances2 = []
    for word1 in common_words:
        for word2 in common_words:
            if word1 != word2:
                dist1 = abs(word_positions1[word1] - word_positions1[word2])
                dist2 = abs(word_positions2[word1] - word_positions2[word2])
                distances1.append(dist1)
                distances2.append(dist2)
    if distances1 and distances2:
        avg_distance_diff = np.mean(
            [abs(d1 - d2) for d1, d2 in zip(distances1, distances2)]
        )
        return avg_distance_diff
    else:
        return 0
