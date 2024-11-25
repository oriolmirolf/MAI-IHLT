# feature_extraction.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string
import math
import spacy
import numpy as np
from collections import Counter
from difflib import SequenceMatcher

# Make sure to download the required NLTK data files and spaCy model:
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')
# For spaCy, install the English model: python -m spacy download en_core_web_sm

nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))


def extract_features(s1, s2):
    """
    Extracts lexical, syntactic, and semantic features.

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        dict: A dictionary of combined features.
    """
    features = {}
    features.update(lexical_features(s1, s2))
    features.update(syntactic_features(s1, s2))
    features.update(semantic_features(s1, s2))
    return features


def lexical_features(s1, s2):
    """
    Extracts lexical features between two sentences.

    Returns:
        dict: A dictionary of lexical features.
    """
    features = {}
    # Preprocess
    tokens1 = word_tokenize(s1.lower())
    tokens2 = word_tokenize(s2.lower())
    tokens1 = [token for token in tokens1 if token not in string.punctuation]
    tokens2 = [token for token in tokens2 if token not in string.punctuation]

    # Features inspired by UKP (Baer et al., 2012) and TakeLab (Sarić et al., 2012)

    # Lexical overlap
    features['lex_word_overlap'] = word_overlap(tokens1, tokens2)

    # N-gram overlaps
    n_values = [1, 2, 3, 4]
    for n in n_values:
        feature_name = f'lex_word_ngram_overlap_{n}'
        overlap = ngram_overlap(tokens1, tokens2, n)
        features[feature_name] = overlap

    # Character n-gram similarity
    n_values = [2, 3, 4, 5]
    for n in n_values:
        feature_name = f'lex_char_ngram_similarity_{n}'
        similarity = char_ngram_similarity(s1.lower(), s2.lower(), n)
        features[feature_name] = similarity

    # Longest Common Substring length
    features['lex_longest_common_substring'] = longest_common_substring(s1.lower(), s2.lower())

    # Longest Common Subsequence length
    features['lex_longest_common_subsequence'] = longest_common_subsequence(tokens1, tokens2)

    # Greedy String Tiling
    features['lex_greedy_string_tiling'] = greedy_string_tiling(s1.lower(), s2.lower())

    # Word n-gram Jaccard similarity
    n_values = [1, 2, 3, 4]
    for n in n_values:
        feature_name = f'lex_word_ngram_jaccard_{n}'
        jaccard = ngram_jaccard(tokens1, tokens2, n)
        features[feature_name] = jaccard

    return features


def word_overlap(tokens1, tokens2):
    """
    Compute word overlap between two token lists.

    Returns:
        float: Word overlap score.
    """
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def ngram_overlap(tokens1, tokens2, n):
    """
    Compute n-gram overlap between two token lists.

    Returns:
        float: N-gram overlap score.
    """
    ngrams1 = list(ngrams(tokens1, n))
    ngrams2 = list(ngrams(tokens2, n))
    set1 = set(ngrams1)
    set2 = set(ngrams2)
    if not set1 or not set2:
        return 0.0

    intersection = set1.intersection(set2)
    overlap1 = len(intersection) / len(set1)
    overlap2 = len(intersection) / len(set2)
    if overlap1 + overlap2 == 0:
        return 0.0
    harmonic_mean = 2 * overlap1 * overlap2 / (overlap1 + overlap2)
    return harmonic_mean


def ngram_jaccard(tokens1, tokens2, n):
    """
    Compute n-gram Jaccard similarity between two token lists.

    Returns:
        float: N-gram Jaccard similarity score.
    """
    ngrams1 = list(ngrams(tokens1, n))
    ngrams2 = list(ngrams(tokens2, n))
    set1 = set(ngrams1)
    set2 = set(ngrams2)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def char_ngram_similarity(s1, s2, n):
    """
    Compute character n-gram similarity between two sentences.

    Returns:
        float: Character n-gram similarity.
    """
    s1 = s1.replace(" ", "")
    s2 = s2.replace(" ", "")
    ngrams1 = [s1[i:i + n] for i in range(len(s1) - n + 1)]
    ngrams2 = [s2[i:i + n] for i in range(len(s2) - n + 1)]
    set1 = set(ngrams1)
    set2 = set(ngrams2)
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union)
    return similarity


def longest_common_substring(s1, s2):
    """
    Compute the length of the longest common substring between two strings.

    Returns:
        float: Normalized length of longest common substring.
    """
    # Implementation inspired by the description in UKP (Baer et al., 2012)
    matcher = SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    length = match.size
    max_length = max(len(s1), len(s2))
    if max_length == 0:
        return 0.0
    return length / max_length


def longest_common_subsequence(tokens1, tokens2):
    """
    Compute the length of the longest common subsequence between two token lists.

    Returns:
        float: Normalized length of longest common subsequence.
    """
    # Implementation inspired by the description in UKP (Baer et al., 2012)
    len1 = len(tokens1)
    len2 = len(tokens2)
    L = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1):
        for j in range(len2):
            if tokens1[i] == tokens2[j]:
                L[i + 1][j + 1] = L[i][j] + 1
            else:
                L[i + 1][j + 1] = max(L[i + 1][j], L[i][j + 1])
    lcs_length = L[len1][len2]
    max_length = max(len1, len2)
    if max_length == 0:
        return 0.0
    return lcs_length / max_length


def greedy_string_tiling(s1, s2, min_match_length=3):
    """
    Compute Greedy String Tiling similarity between two strings.
    This function approximates the Greedy String Tiling algorithm.

    Returns:
        float: GST similarity score.
    """
    # Simplified implementation inspired by UKP (Baer et al., 2012)
    matches = []
    remaining_s1 = s1
    remaining_s2 = s2
    total_matches = 0
    while True:
        matcher = SequenceMatcher(None, remaining_s1, remaining_s2)
        match = matcher.find_longest_match(0, len(remaining_s1), 0, len(remaining_s2))
        if match.size < min_match_length:
            break
        total_matches += match.size
        # Mask matched substrings
        remaining_s1 = remaining_s1[:match.a] + ' ' * match.size + remaining_s1[match.a + match.size:]
        remaining_s2 = remaining_s2[:match.b] + ' ' * match.size + remaining_s2[match.b + match.size:]
    normalized_similarity = total_matches / (len(s1) + len(s2))
    return normalized_similarity


def syntactic_features(s1, s2):
    """
    Extracts syntactic features between two sentences.

    Returns:
        dict: A dictionary of syntactic features.
    """
    features = {}
    # POS tagging
    tokens1 = word_tokenize(s1)
    tokens2 = word_tokenize(s2)
    pos_tags1 = nltk.pos_tag(tokens1)
    pos_tags2 = nltk.pos_tag(tokens2)
    pos1 = [pos for word, pos in pos_tags1]
    pos2 = [pos for word, pos in pos_tags2]

    # POS n-gram overlaps
    n_values = [1, 2, 3, 4]
    for n in n_values:
        feature_name = f'syn_pos_ngram_overlap_{n}'
        overlap = ngram_overlap(pos1, pos2, n)
        features[feature_name] = overlap

    # Syntactic dependency overlap (TakeLab, Sarić et al., 2012)
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    dep_triples1 = extract_dep_triples(doc1)
    dep_triples2 = extract_dep_triples(doc2)
    dep_overlap = dependency_overlap(dep_triples1, dep_triples2)
    features['syn_dependency_overlap'] = dep_overlap

    # Syntactic role similarity (TakeLab, Sarić et al., 2012)
    roles_similarity = syntactic_role_similarity(doc1, doc2)
    features.update(roles_similarity)

    # Sentence length difference
    len_diff = abs(len(tokens1) - len(tokens2)) / ((len(tokens1) + len(tokens2))/2)
    features['syn_length_difference'] = len_diff

    return features


def extract_dep_triples(doc):
    """
    Extract dependency triples from a spaCy doc.

    Returns:
        set: Set of dependency triples.
    """
    dep_triples = set()
    for token in doc:
        if token.dep_ == 'ROOT':
            continue
        triple = (token.head.text, token.dep_, token.text)
        dep_triples.add(triple)
    return dep_triples


def dependency_overlap(triples1, triples2):
    """
    Compute overlap between two sets of dependency triples.

    Returns:
        float: Dependency overlap score.
    """
    if not triples1 or not triples2:
        return 0.0
    intersection = triples1.intersection(triples2)
    overlap1 = len(intersection) / len(triples1)
    overlap2 = len(intersection) / len(triples2)
    if overlap1 + overlap2 == 0:
        return 0.0
    else:
        harmonic_mean = 2 * overlap1 * overlap2 / (overlap1 + overlap2)
        return harmonic_mean


def syntactic_role_similarity(doc1, doc2):
    """
    Compute syntactic role similarities between two spaCy docs.

    Returns:
        dict: Dictionary of syntactic role similarity features.
    """
    features = {}
    roles = ['nsubj', 'dobj', 'iobj', 'pobj', 'ROOT']
    for role in roles:
        tokens1 = [token for token in doc1 if token.dep_ == role]
        tokens2 = [token for token in doc2 if token.dep_ == role]

        # Compute similarity using WordNet path similarity
        sim = 0.0
        count = 0
        for token1 in tokens1:
            for token2 in tokens2:
                sim_value = wordnet_similarity(token1.lemma_, token2.lemma_)
                if sim_value is not None:
                    sim += sim_value
                    count += 1
        if count > 0:
            sim /= count
        features[f'syn_role_similarity_{role}'] = sim
        features[f'syn_role_exists_{role}'] = int(bool(tokens1) and bool(tokens2))
    return features


def semantic_features(s1, s2):
    """
    Extracts semantic features between two sentences.

    Returns:
        dict: A dictionary of semantic features.
    """
    features = {}
    # Tokenization and lemmatization
    tokens1 = word_tokenize(s1.lower())
    tokens2 = word_tokenize(s2.lower())
    tokens1 = [token for token in tokens1 if token not in string.punctuation]
    tokens2 = [token for token in tokens2 if token not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    lemmas1 = [lemmatizer.lemmatize(token) for token in tokens1]
    lemmas2 = [lemmatizer.lemmatize(token) for token in tokens2]

    # Weighted word overlap (TakeLab, Sarić et al., 2012)
    weighted_overlap = weighted_word_overlap(lemmas1, lemmas2)
    features['sem_weighted_word_overlap'] = weighted_overlap

    # Greedy lemma alignment overlap using WordNet similarity (TakeLab, Sarić et al., 2012)
    sim_score = greedy_lemma_align_overlap(lemmas1, lemmas2)
    features['sem_greedy_lemma_align_overlap'] = sim_score

    # Pairwise word similarity using WordNet (UKP, Baer et al., 2012)
    pairwise_similarity = pairwise_word_similarity(lemmas1, lemmas2)
    features['sem_pairwise_word_similarity'] = pairwise_similarity

    # Sentence embeddings similarity (We can use simple averaging of word embeddings)
    # Note: Since pre-trained embeddings like BERT are not allowed, we may skip this or use other embeddings.

    return features


def weighted_word_overlap(lemmas1, lemmas2):
    """
    Compute weighted word overlap between two lists of lemmas.

    Returns:
        float: Weighted word overlap score.
    """
    # Use information content (IC) from word frequencies.
    # For simplicity, we can approximate IC with inverse document frequency (IDF).
    # Since we don't have a corpus to compute IDF, let's assign a default value.

    # For this implementation, we'll assign all words an IC of 1.
    # In real implementation, use a corpus to compute IC.

    lemmas1_set = set(lemmas1)
    lemmas2_set = set(lemmas2)
    common_lemmas = lemmas1_set.intersection(lemmas2_set)
    if not lemmas1_set or not lemmas2_set:
        return 0.0
    numerator = sum(1 for lemma in common_lemmas)
    denominator = sum(1 for lemma in lemmas1_set.union(lemmas2_set))
    return numerator / denominator


def pairwise_word_similarity(lemmas1, lemmas2):
    """
    Compute average of maximum pairwise word similarities between two lists of lemmas.

    Returns:
        float: Average maximum similarity score.
    """
    max_similarities = []
    for lemma1 in lemmas1:
        max_sim = 0.0
        synsets1 = wordnet.synsets(lemma1)
        if not synsets1:
            continue
        for lemma2 in lemmas2:
            synsets2 = wordnet.synsets(lemma2)
            if not synsets2:
                continue
            sim = max((syn1.wup_similarity(syn2) or 0) for syn1 in synsets1 for syn2 in synsets2)
            if sim > max_sim:
                max_sim = sim
        if max_sim > 0:
            max_similarities.append(max_sim)
    if not max_similarities:
        return 0.0
    return sum(max_similarities) / len(max_similarities)


def wordnet_similarity(word1, word2):
    """
    Compute WordNet path similarity between two words.

    Returns:
        float: Similarity score.
    """
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    if not synsets1 or not synsets2:
        return None
    max_sim = max((syn1.path_similarity(syn2) or 0) for syn1 in synsets1 for syn2 in synsets2)
    return max_sim


def greedy_lemma_align_overlap(lemmas1, lemmas2):
    """
    Compute greedy lemma alignment overlap using WordNet similarity.

    Returns:
        float: Alignment overlap score.
    """
    # Similar to TakeLab's greedy lemma alignment (Sarić et al., 2012)
    similarity_matrix = {}
    for i, lemma1 in enumerate(lemmas1):
        synsets1 = wordnet.synsets(lemma1)
        if not synsets1:
            continue
        for j, lemma2 in enumerate(lemmas2):
            synsets2 = wordnet.synsets(lemma2)
            if not synsets2:
                continue
            max_similarity = max((syn1.wup_similarity(syn2) or 0) for syn1 in synsets1 for syn2 in synsets2)
            if max_similarity > 0:
                similarity_matrix[(i, j)] = max_similarity

    # Greedy alignment
    aligned_pairs = []
    used_i = set()
    used_j = set()
    for (i, j), sim in sorted(similarity_matrix.items(), key=lambda item: -item[1]):
        if i not in used_i and j not in used_j:
            aligned_pairs.append((i, j, sim))
            used_i.add(i)
            used_j.add(j)
    sim_sum = sum(sim for i, j, sim in aligned_pairs)
    max_possible = max(len(lemmas1), len(lemmas2))
    if max_possible == 0:
        return 0.0
    else:
        return sim_sum / max_possible

