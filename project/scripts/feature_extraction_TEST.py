# feature_extraction.py

import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.util import ngrams
from nltk.corpus import wordnet as wn
from nltk.metrics.distance import jaro_winkler_similarity
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet_ic

import spacy
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from itertools import chain
import functools

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import pandas as pd

# Download required NLTK data files
nltk.download('wordnet')
nltk.download('wordnet_ic')
nltk.download('sentiwordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load the information content file for Resnik similarity
brown_ic = wordnet_ic.ic('ic-brown.dat')

# ---------------------------- #
#  FEATURE EXTRACTOR FUNCTION  #
# ---------------------------- #

class FeatureExtractor:
    def __init__(self):
        """Initialize the FeatureExtractor with an internal memoization dictionary."""
        self.memoized_pair_features = {}

    def extract_pair_features(self, s1, s2):
        """Extract features for a pair of sentences with order-invariant memoization."""
        # Use frozenset to make memoization invariant to the order of s1 and s2
        key = frozenset([s1, s2])
        if key in self.memoized_pair_features:
            return self.memoized_pair_features[key]
        features = extract_features(s1, s2)  # Custom feature computation
        self.memoized_pair_features[key] = features
        return features

    def process_row(self, row):
        """Process a single row and extract features for the sentence pair."""
        s1, s2, score, ds = row

        # Extract features for the sentence pair
        features = self.extract_pair_features(s1, s2)

        # Include score and dataset in the result
        features['score'] = score
        features['dataset'] = ds

        return features

    def extract_features_parallel(self, data):
        """Extract features in parallel and return a DataFrame."""
        with Pool(cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(self.process_row, data),
                    total=len(data),
                    desc="Extracting Features"
                )
            )
        # Convert the list of feature dictionaries to a DataFrame
        return pd.DataFrame(results)
    
    def extract_features_sequential(self, data):
        """Extract features sequentially and return a DataFrame."""
        results = []
        for row in tqdm(data, desc="Extracting Features (Sequential)"):
            results.append(self.process_row(row))
        # Convert the list of feature dictionaries to a DataFrame
        return pd.DataFrame(results)

# ----------------------- #
# ALL RELEVANT FUNCTIONS  #
# ----------------------- #

@functools.lru_cache(maxsize=None)
def preprocess_sentence(s):
    """
    Preprocess a sentence by tokenizing and lowercasing.

    Returns:
        tuple: A tuple of tokens.
    """
    tokens = nltk.word_tokenize(s.lower())
    return tuple(tokens)

@functools.lru_cache(maxsize=None)
def get_pos_tags(tokens):
    """
    Get POS tags for a list of tokens.

    Returns:
        tuple: A tuple of (word, POS tag) pairs.
    """
    pos_tags = nltk.pos_tag(tokens)
    return tuple(pos_tags)

@functools.lru_cache(maxsize=None)
def get_dependency_relations(s):
    """
    Get dependency relations from a sentence using spaCy.

    Returns:
        tuple: A tuple of dependency relations.
    """
    doc = nlp(s)
    deps = []
    for token in doc:
        if token.dep_ != 'ROOT':
            deps.append((token.dep_, token.head.text, token.text))
    return tuple(deps)

@functools.lru_cache(maxsize=None)
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
    Compute lexical similarity features between two sentences.

    Derived from methods used in SemEval 2012 papers [Bär et al., 2012], [Štajner et al., 2012], [Glinos, 2012], [UKP at SemEval-2012].

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        dict: A dictionary of lexical features.
    """
    tokens1 = preprocess_sentence(s1)
    tokens2 = preprocess_sentence(s2)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens1_no_stop = [w for w in tokens1 if w not in stopwords]
    tokens2_no_stop = [w for w in tokens2 if w not in stopwords]

    # Word overlap
    # Reference: Basic text similarity measures used by multiple teams in SemEval 2012
    overlap = set(tokens1_no_stop).intersection(set(tokens2_no_stop))
    union = set(tokens1_no_stop).union(set(tokens2_no_stop))
    word_overlap_ratio = len(overlap) / len(union) if len(union) != 0 else 0

    # Jaccard similarity
    # Reference: Used by multiple teams in SemEval 2012 [Bär et al., 2012], [Glinos, 2012]
    jaccard = word_overlap_ratio

    # Dice coefficient
    # Reference: Used by teams in SemEval 2012 [Bär et al., 2012]
    dice_coeff = (2 * len(overlap)) / (len(tokens1_no_stop) + len(tokens2_no_stop)) if (len(tokens1_no_stop) + len(tokens2_no_stop)) != 0 else 0

    # Overlap coefficient
    # Reference: Basic text similarity measures
    min_len = min(len(tokens1_no_stop), len(tokens2_no_stop))
    overlap_coeff = len(overlap) / min_len if min_len != 0 else 0

    # Levenshtein Distance (Edit Distance)
    # Reference: Used by [Glinos, 2012]
    edit_distance = nltk.edit_distance(' '.join(tokens1), ' '.join(tokens2))
    max_len = max(len(' '.join(tokens1)), len(' '.join(tokens2)))
    norm_edit_distance = 1 - (edit_distance / max_len) if max_len != 0 else 0

    # Jaro-Winkler similarity
    # Reference: Used by [Jimenez et al., 2012], [UKP at SemEval-2012]
    jaro_winkler = jaro_winkler_similarity(' '.join(tokens1), ' '.join(tokens2))

    # Cosine similarity using TF-IDF vectors
    # Reference: Used by the UKP team [Bär et al., 2012]
    tfidf_vectorizer = TfidfVectorizer().fit([s1, s2])
    tfidf_vectors = tfidf_vectorizer.transform([s1, s2])
    cosine_sim = cosine_similarity(tfidf_vectors[0], tfidf_vectors[1])[0][0]

    # Euclidean distance between TF-IDF vectors
    # Reference: Basic vector space models
    euclidean_dist = np.linalg.norm(tfidf_vectors[0].toarray() - tfidf_vectors[1].toarray())

    # Character n-gram overlap (n=2,3,4)
    # Reference: Used by [Jimenez et al., 2012], [Bär et al., 2012], [UKP at SemEval-2012]
    char_ngram_overlaps = {}
    for n in [2, 3, 4]:
        char_ngrams1 = set([''.join(gram) for token in tokens1 for gram in ngrams(token, n)])
        char_ngrams2 = set([''.join(gram) for token in tokens2 for gram in ngrams(token, n)])
        char_overlap = len(char_ngrams1.intersection(char_ngrams2))
        char_union = len(char_ngrams1.union(char_ngrams2))
        char_ngram_overlap = char_overlap / char_union if char_union != 0 else 0
        char_ngram_overlaps[f'lex_char_{n}gram_overlap'] = char_ngram_overlap

    # Character n-gram TF-IDF cosine similarity (n=2 to 4)
    # Reference: Character-level features used by [Bär et al., 2012], [UKP at SemEval-2012]
    char_tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4)).fit([s1, s2])
    char_tfidf_vectors = char_tfidf_vectorizer.transform([s1, s2])
    char_tfidf_cosine_sim = cosine_similarity(char_tfidf_vectors[0], char_tfidf_vectors[1])[0][0]

    # Word n-gram overlap using Containment measure (n=1,2), without stopwords
    # Reference: Used by [UKP at SemEval-2012], Containment measure (Broder, 1997)
    word_ngram_containments = {}
    for n in [1, 2]:
        ngrams1 = set(ngrams(tokens1_no_stop, n))
        ngrams2 = set(ngrams(tokens2_no_stop, n))
        intersection = ngrams1.intersection(ngrams2)
        containment = len(intersection) / min(len(ngrams1), len(ngrams2)) if min(len(ngrams1), len(ngrams2)) != 0 else 0
        word_ngram_containments[f'lex_word_{n}gram_containment'] = containment

    # Word n-gram overlap using Jaccard coefficient (n=1 to 4), with and without stopwords
    # Reference: Used by [UKP at SemEval-2012]
    word_ngram_jaccards = {}
    for n in [1, 2, 3, 4]:
        for stopword_setting, tokens1_set, tokens2_set in [('with_stop', tokens1, tokens2), ('no_stop', tokens1_no_stop, tokens2_no_stop)]:
            ngrams1 = set(ngrams(tokens1_set, n))
            ngrams2 = set(ngrams(tokens2_set, n))
            intersection = ngrams1.intersection(ngrams2)
            union = ngrams1.union(ngrams2)
            jaccard = len(intersection) / len(union) if len(union) != 0 else 0
            word_ngram_jaccards[f'lex_word_{n}gram_jaccard_{stopword_setting}'] = jaccard

    # Longest Common Substring length normalized by average sentence length
    # Reference: Used by [Bär et al., 2012], [Glinos, 2012], [UKP at SemEval-2012]
    lcs_length = longest_common_substring_length(' '.join(tokens1), ' '.join(tokens2))
    avg_len = (len(' '.join(tokens1)) + len(' '.join(tokens2))) / 2
    lcs_norm = lcs_length / avg_len if avg_len != 0 else 0

    # Greedy String Tiling similarity
    # Reference: Used by [UKP at SemEval-2012], Greedy String Tiling (Wise, 1996)
    gst_similarity = greedy_string_tiling(' '.join(tokens1), ' '.join(tokens2), min_match_length=3)

    # Existing features retained
    # Word n-gram overlap (n=2)
    ngram_overlap = ngram_overlap_ratio(tokens1_no_stop, tokens2_no_stop, n=2)

    # BLEU score
    # Reference: Used by [Bär et al., 2012]
    bleu_score = nltk.translate.bleu_score.sentence_bleu(
        [tokens1_no_stop], tokens2_no_stop,
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)

    # Common words count
    common_word_count = len(overlap)

    # Total unique words count
    unique_word_count = len(union)

    # Ratio of common words to total unique words
    common_to_unique_ratio = common_word_count / unique_word_count if unique_word_count != 0 else 0

    # Length ratio
    # Reference: Used by [Bär et al., 2012]
    len_ratio = min(len(tokens1), len(tokens2)) / max(len(tokens1), len(tokens2)) if max(len(tokens1), len(tokens2)) != 0 else 0

    # Absolute length difference
    length_diff = abs(len(tokens1) - len(tokens2))

    # Content word overlap ratio (nouns, verbs, adjectives, adverbs)
    # Reference: Content word overlap used in similarity measures [Jimenez et al., 2012]
    pos_tags1_full = get_pos_tags(tokens1)
    pos_tags2_full = get_pos_tags(tokens2)
    content_pos_tags = ('NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                        'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
                        'JJ', 'JJR', 'JJS',  # Adjectives
                        'RB', 'RBR', 'RBS')  # Adverbs
    tokens1_content = [w for w, pos in pos_tags1_full if pos in content_pos_tags]
    tokens2_content = [w for w, pos in pos_tags2_full if pos in content_pos_tags]

    content_overlap = set(tokens1_content).intersection(set(tokens2_content))
    content_union = set(tokens1_content).union(set(tokens2_content))
    content_word_overlap_ratio = len(content_overlap) / len(content_union) if len(content_union) != 0 else 0

    # Word n-gram precision and recall (n=1 to 3)
    ngram_precisions = []
    ngram_recalls = []
    for n in range(1, 4):
        ngrams1 = list(ngrams(tokens1_no_stop, n))
        ngrams2 = list(ngrams(tokens2_no_stop, n))
        ngrams1_set = set(ngrams1)
        ngrams2_set = set(ngrams2)
        overlap = ngrams1_set.intersection(ngrams2_set)
        total_ngrams1 = len(ngrams1)
        total_ngrams2 = len(ngrams2)
        precision = len(overlap) / total_ngrams2 if total_ngrams2 != 0 else 0
        recall = len(overlap) / total_ngrams1 if total_ngrams1 != 0 else 0
        ngram_precisions.append(precision)
        ngram_recalls.append(recall)

    # Soft Cardinality Similarity (Lexical)
    # Reference: [Jimenez et al., 2012] - UNAL-NLP: Combining Soft Cardinality Features
    tokens1_no_stop_set = set(tokens1_no_stop)
    tokens2_no_stop_set = set(tokens2_no_stop)
    soft_jaccard = soft_jaccard_similarity(tokens1_no_stop_set, tokens2_no_stop_set)

    features.update({
        'lex_jaccard': jaccard,
        'lex_dice_coeff': dice_coeff,
        'lex_overlap_coeff': overlap_coeff,
        'lex_norm_edit_distance': norm_edit_distance,
        'lex_jaro_winkler': jaro_winkler,
        'lex_cosine_sim': cosine_sim,
        'lex_euclidean_dist': euclidean_dist,
        'lex_lcs_norm': lcs_norm,
        'lex_gst_similarity': gst_similarity,
        'lex_ngram_overlap': ngram_overlap,
        'lex_bleu_score': bleu_score,
        'lex_common_word_count': common_word_count,
        'lex_common_to_unique_ratio': common_to_unique_ratio,
        'lex_len_ratio': len_ratio,
        'lex_length_diff': length_diff,
        'lex_content_word_overlap_ratio': content_word_overlap_ratio,
        'lex_ngram_precision_1': ngram_precisions[0],
        'lex_ngram_precision_2': ngram_precisions[1],
        'lex_ngram_precision_3': ngram_precisions[2],
        'lex_ngram_recall_1': ngram_recalls[0],
        'lex_ngram_recall_2': ngram_recalls[1],
        'lex_ngram_recall_3': ngram_recalls[2],
        'lex_soft_jaccard': soft_jaccard,
    })
    features.update(char_ngram_overlaps)
    features.update(word_ngram_containments)
    features.update(word_ngram_jaccards)

    return features

def syntactic_features(s1, s2):
    """
    Compute syntactic similarity features between two sentences.

    Derived from methods in SemEval 2012 papers [Bär et al., 2012], [Štajner et al., 2012], [Glinos, 2012], [Croce et al., 2012], [UKP at SemEval-2012].

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        dict: A dictionary of syntactic features.
    """
    tokens1 = preprocess_sentence(s1)
    tokens2 = preprocess_sentence(s2)

    pos_tags1_full = get_pos_tags(tokens1)
    pos_tags2_full = get_pos_tags(tokens2)

    pos_tags1 = [pos for _, pos in pos_tags1_full]
    pos_tags2 = [pos for _, pos in pos_tags2_full]

    # POS tag overlap ratio
    # Reference: Used by [Bär et al., 2012]
    pos_overlap = set(pos_tags1).intersection(set(pos_tags2))
    avg_pos_length = (len(pos_tags1) + len(pos_tags2)) / 2
    pos_overlap_ratio = len(pos_overlap) / avg_pos_length if avg_pos_length != 0 else 0

    # POS tag bigram overlap ratio
    # Reference: Used by [Bär et al., 2012]
    pos_bigram_overlap = ngram_overlap_ratio(pos_tags1, pos_tags2, n=2)

    # POS tag trigram overlap ratio
    # Reference: Used by [Bär et al., 2012]
    pos_trigram_overlap = ngram_overlap_ratio(pos_tags1, pos_tags2, n=3)

    # POS tag n-gram overlap using Containment measure (n=2 to 4)
    # Reference: Used by [UKP at SemEval-2012], Containment measure
    pos_ngram_containments = {}
    for n in [2, 3, 4]:
        ngrams1 = set(ngrams(pos_tags1, n))
        ngrams2 = set(ngrams(pos_tags2, n))
        intersection = ngrams1.intersection(ngrams2)
        containment = len(intersection) / min(len(ngrams1), len(ngrams2)) if min(len(ngrams1), len(ngrams2)) != 0 else 0
        pos_ngram_containments[f'syn_pos_{n}gram_containment'] = containment

    # Dependency relation overlap
    # Reference: Used by [Štajner et al., 2012]
    deps1 = get_dependency_relations(s1)
    deps2 = get_dependency_relations(s2)
    dep_relations1 = set([dep[0] for dep in deps1])
    dep_relations2 = set([dep[0] for dep in deps2])
    dep_overlap = dep_relations1.intersection(dep_relations2)
    dep_union = dep_relations1.union(dep_relations2)
    dep_overlap_ratio = len(dep_overlap) / len(dep_union) if len(dep_union) != 0 else 0

    # Grammatical relations proportions
    # Reference: Used by [Bär et al., 2012]
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
    # Reference: Used by [Bär et al., 2012]
    word_order_sim = word_order_similarity(tokens1, tokens2)

    # Longest common subsequence
    # Reference: Used by [Bär et al., 2012], [Glinos, 2012]
    avg_length = (len(tokens1) + len(tokens2)) / 2
    lcs_length = longest_common_subsequence(tokens1, tokens2)
    lcs_norm = lcs_length / avg_length if avg_length != 0 else 0

    # Tree edit distance (simplified)
    # Reference: Used by [Štajner et al., 2012]
    tree_edit_dist = tree_edit_distance(s1, s2)

    # POS tag sequence similarity (normalized edit distance)
    # Reference: Used by [Glinos, 2012]
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
    # Reference: Considering POS tag distributions as features, inspired by [Glinos, 2012]
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
    }
    features.update(pos_diff_features)
    features.update(pos_ngram_containments)
    return features

def semantic_features(s1, s2):
    """
    Compute semantic similarity features between two sentences.

    Features inspired by methods used in SemEval 2012 Task 6 participant papers, including UKP at SemEval-2012.

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        dict: A dictionary of semantic features.
    """
    tokens1 = preprocess_sentence(s1)
    tokens2 = preprocess_sentence(s2)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens1_no_stop = [w for w in tokens1 if w not in stopwords]
    tokens2_no_stop = [w for w in tokens2 if w not in stopwords]

    # Word Similarity using Resnik (1995) on WordNet aggregated according to Mihalcea et al. (2006)
    # Reference: Used by [UKP at SemEval-2012]
    resnik_similarity = word_similarity_resnik_aggregate(tokens1_no_stop, tokens2_no_stop)

    # Explicit Semantic Analysis (ESA) similarity
    # Reference: Used by [UKP at SemEval-2012]
    # Approximate ESA using pre-trained embeddings
    esa_similarity = esa_sim(tokens1_no_stop, tokens2_no_stop)

    # Synonym Matching Based on WordNet Synsets
    # Reference: Used by [Bär et al., 2012], [Jimenez et al., 2012]
    synonym_overlap = synonym_overlap_ratio(tokens1, tokens2)

    # Lexical Chain Overlap
    # Reference: Used by [Bär et al., 2012]
    lexical_chain_overlap = lexical_chain_overlap_ratio(tokens1, tokens2)

    # WordNet-based similarity measures
    # Reference: Used by [UKP at SemEval-2012]

    # Define similarity functions for WordNet
    def avg_max_wordnet_similarity(tokens1, tokens2, similarity_func):
        """
        Compute the average and maximum word similarity between two lists of tokens using WordNet and given similarity function.

        Args:
            tokens1 (list): First list of tokens.
            tokens2 (list): Second list of tokens.
            similarity_func: Function to compute similarity between two synsets.

        Returns:
            avg_sim: Average maximum similarity for tokens in tokens1.
            max_sim: Maximum similarity across all token pairs.
        """
        total_sim = 0
        count = 0
        max_sim = 0

        for word1 in tokens1:
            synsets1 = wn.synsets(word1)
            for word2 in tokens2:
                synsets2 = wn.synsets(word2)
                for syn1 in synsets1:
                    for syn2 in synsets2:
                        if syn1.pos() == syn2.pos():
                            sim = similarity_func(syn1, syn2)
                            if sim is not None:
                                if sim > max_sim:
                                    max_sim = sim
                                total_sim += sim
                                count += 1

        avg_sim = total_sim / count if count > 0 else 0
        return avg_sim, max_sim

    # WordNet Path Similarity (Average and Max)
    avg_sim_path, max_sim_path = avg_max_wordnet_similarity(tokens1, tokens2, lambda s1, s2: s1.path_similarity(s2))

    # WordNet Wu-Palmer Similarity (Average and Max)
    avg_sim_wup, max_sim_wup = avg_max_wordnet_similarity(tokens1, tokens2, lambda s1, s2: s1.wup_similarity(s2))

    # WordNet Leacock-Chodorow Similarity (Average and Max)
    avg_sim_lch, max_sim_lch = avg_max_wordnet_similarity(tokens1, tokens2, lambda s1, s2: s1.lch_similarity(s2))

    # Antonym overlap ratio
    # Reference: Capturing antonyms and negation differences, inspired by [Bär et al., 2012]
    antonym_ratio = antonym_overlap(tokens1, tokens2)

    # Named entity overlap
    # Reference: Used by [Bär et al., 2012]
    ne_overlap = named_entity_overlap(s1, s2)
    ne_type_overlap = named_entity_type_overlap(s1, s2)

    # Simplified Lesk-based similarity
    # Reference: Used by [Bär et al., 2012]
    simplified_lesk_sim = simplified_lesk_similarity(tokens1, tokens2)

    # Hypernym/Hyponym Overlap
    # Reference: Used by [Bär et al., 2012], [Štajner et al., 2012]
    hypernym_hyponym_overlap = hypernym_hyponym_overlap_ratio(tokens1, tokens2)

    # LSA and LDA similarities
    # Reference: Used by [Bär et al., 2012]
    lsa_sim = lsa_similarity(s1, s2)
    lda_sim = lda_similarity(s1, s2)

    # Sentiment scores and differences using SentiWordNet
    # Reference: Incorporating sentiment features as in [Gupta et al., 2012]
    pos1, neg1, obj1 = compute_sentiment_score(tokens1)
    pos2, neg2, obj2 = compute_sentiment_score(tokens2)

    sentiment_diff = {
        'sem_sentiment_pos_diff': abs(pos1 - pos2),
        'sem_sentiment_neg_diff': abs(neg1 - neg2),
        'sem_sentiment_obj_diff': abs(obj1 - obj2),
    }

    # Negation features
    # Reference: Considering negation as an important factor, as in [Bär et al., 2012]
    neg_count1 = count_negations(tokens1)
    neg_count2 = count_negations(tokens2)
    negation_feature = {
        'sem_negation_difference': abs(neg_count1 - neg_count2),
        'sem_negation_both_present': int(neg_count1 > 0 and neg_count2 > 0),
        'sem_negation_both_absent': int(neg_count1 == 0 and neg_count2 == 0)
    }

    # Semantic Role Labeling (SRL) Overlap
    # Reference: [Heilman and Madnani, 2012] - ETS: Discriminative Edit Models
    sem_srl_overlap = semantic_role_overlap(s1, s2)

    # Temporal Expression Overlap
    # Reference: [Sultan et al., 2012] - DLS@CU: Sentence Similarity from Word Alignment
    sem_temporal_overlap = temporal_expression_overlap(s1, s2)

    features = {
        'sem_resnik_similarity': resnik_similarity,
        'sem_esa_similarity': esa_similarity,
        'sem_synonym_overlap': synonym_overlap,
        'sem_lexical_chain_overlap': lexical_chain_overlap,
        'sem_ne_overlap': ne_overlap,
        'sem_ne_type_overlap': ne_type_overlap,
        'sem_simplified_lesk_sim': simplified_lesk_sim,
        'sem_hypernym_hyponym_overlap': hypernym_hyponym_overlap,
        'sem_lsa_sim': lsa_sim,
        'sem_lda_sim': lda_sim,
        'sem_avg_sim_path': avg_sim_path,
        'sem_max_sim_path': max_sim_path,
        'sem_avg_sim_wup': avg_sim_wup,
        'sem_max_sim_wup': max_sim_wup,
        'sem_avg_sim_lch': avg_sim_lch,
        'sem_max_sim_lch': max_sim_lch,
        'sem_antonym_ratio': antonym_ratio,
        'sem_srl_overlap': sem_srl_overlap,
        'sem_temporal_overlap': sem_temporal_overlap,
    }
    features.update(sentiment_diff)
    features.update(negation_feature)
    return features

def word_similarity_resnik_aggregate(tokens1, tokens2):
    """
    Compute word similarity using Resnik (1995) on WordNet aggregated according to Mihalcea et al. (2006).

    Args:
        tokens1 (list): Tokens from the first sentence (without stopwords).
        tokens2 (list): Tokens from the second sentence (without stopwords).

    Returns:
        float: Aggregated Resnik similarity score.
    """
    # Build IDF weights using a dummy corpus (as we don't have a large corpus here)
    # For this example, we'll assign an IDF of 1 to all words
    idf_weights = {word: 1.0 for word in set(tokens1 + tokens2)}

    # Function to compute maximum Resnik similarity for a word
    def max_resnik_similarity(word, other_tokens):
        max_sim = 0
        synsets1 = wn.synsets(word)
        for other_word in other_tokens:
            synsets2 = wn.synsets(other_word)
            for s1 in synsets1:
                for s2 in synsets2:
                    if s1.pos() == s2.pos():
                        sim = s1.res_similarity(s2, brown_ic)
                        if sim is not None and sim > max_sim:
                            max_sim = sim
        return max_sim

    # Compute aggregated similarity from s1 to s2
    sim_s1_s2 = sum(idf_weights[word] * max_resnik_similarity(word, tokens2) for word in tokens1)
    sim_s2_s1 = sum(idf_weights[word] * max_resnik_similarity(word, tokens1) for word in tokens2)
    # Normalize by the sum of IDF weights
    sum_idf_s1 = sum(idf_weights[word] for word in tokens1)
    sum_idf_s2 = sum(idf_weights[word] for word in tokens2)
    if sum_idf_s1 + sum_idf_s2 > 0:
        sim = (sim_s1_s2 + sim_s2_s1) / (sum_idf_s1 + sum_idf_s2)
    else:
        sim = 0.0
    return sim

def esa_sim(tokens1, tokens2):
    """
    Approximate Explicit Semantic Analysis (ESA) similarity using pre-trained word embeddings.

    Args:
        tokens1 (list): Tokens from the first sentence (without stopwords).
        tokens2 (list): Tokens from the second sentence (without stopwords).

    Returns:
        float: ESA similarity score.
    """
    # Use spaCy's GloVe embeddings as an approximation
    doc1 = nlp(' '.join(tokens1))
    doc2 = nlp(' '.join(tokens2))
    if doc1.vector_norm and doc2.vector_norm:
        sim = doc1.similarity(doc2)
    else:
        sim = 0.0
    return sim

def greedy_string_tiling(s1, s2, min_match_length=3):
    """
    Compute the Greedy String Tiling similarity between two strings.

    Reference: Wise, M. J. (1996). YAP3: Improved detection of similarities in computer program and other texts.

    Args:
        s1 (str): First string.
        s2 (str): Second string.
        min_match_length (int): Minimum match length.

    Returns:
        float: Greedy String Tiling similarity score.
    """
    # Tokenize the strings
    tokens1 = s1.split()
    tokens2 = s2.split()
    matches = []
    marked1 = [False] * len(tokens1)
    marked2 = [False] * len(tokens2)
    max_match_length = min(len(tokens1), len(tokens2))

    total_matched = 0

    for match_length in range(max_match_length, min_match_length - 1, -1):
        i = 0
        while i <= len(tokens1) - match_length:
            if not any(marked1[i:i + match_length]):
                substring1 = tokens1[i:i + match_length]
                j = 0
                while j <= len(tokens2) - match_length:
                    if not any(marked2[j:j + match_length]):
                        substring2 = tokens2[j:j + match_length]
                        if substring1 == substring2:
                            # Mark the tokens
                            for k in range(match_length):
                                marked1[i + k] = True
                                marked2[j + k] = True
                            total_matched += match_length
                            i += match_length - 1  # Skip over the matched tokens
                            break
                    j += 1
            i += 1
    average_length = (len(tokens1) + len(tokens2)) / 2
    gst_similarity = total_matched / average_length if average_length != 0 else 0
    return gst_similarity

# Existing functions for other features remain unchanged

def synonym_overlap_ratio(tokens1, tokens2):
    """
    Calculate synonym overlap ratio using WordNet synonyms.
    """
    synonyms1 = set()
    synonyms2 = set()

    for token in tokens1:
        synsets = wn.synsets(token)
        synonyms1.update(set(chain.from_iterable([syn.lemma_names() for syn in synsets])))

    for token in tokens2:
        synsets = wn.synsets(token)
        synonyms2.update(set(chain.from_iterable([syn.lemma_names() for syn in synsets])))

    overlap = synonyms1.intersection(synonyms2)
    union = synonyms1.union(synonyms2)
    return len(overlap) / len(union) if len(union) else 0

def lexical_chain_overlap_ratio(tokens1, tokens2):
    """
    Calculate the overlap ratio of lexical chains between two token lists.
    Lexical chains are constructed using hypernyms and hyponyms.
    """
    chain1 = set()
    chain2 = set()
    for token in tokens1:
        synsets = wn.synsets(token)
        for syn in synsets:
            chain1.update(set(syn.lemma_names()))
            hypernyms = syn.hypernyms()
            hyponyms = syn.hyponyms()
            for h in hypernyms + hyponyms:
                chain1.update(set(h.lemma_names()))

    for token in tokens2:
        synsets = wn.synsets(token)
        for syn in synsets:
            chain2.update(set(syn.lemma_names()))
            hypernyms = syn.hypernyms()
            hyponyms = syn.hyponyms()
            for h in hypernyms + hyponyms:
                chain2.update(set(h.lemma_names()))

    overlap = chain1.intersection(chain2)
    union = chain1.union(chain2)
    return len(overlap) / len(union) if len(union) else 0

def named_entity_overlap(s1, s2):
    """
    Calculate named entity overlap between two sentences.
    """
    doc1 = nlp(s1)
    doc2 = nlp(s2)

    ne1 = set([ent.text.lower() for ent in doc1.ents])
    ne2 = set([ent.text.lower() for ent in doc2.ents])

    overlap = ne1.intersection(ne2)
    union = ne1.union(ne2)
    return len(overlap) / len(union) if len(union) else 0

def named_entity_type_overlap(s1, s2):
    """
    Calculate named entity type overlap between two sentences.
    """
    doc1 = nlp(s1)
    doc2 = nlp(s2)

    ne_types1 = set([(ent.text.lower(), ent.label_) for ent in doc1.ents])
    ne_types2 = set([(ent.text.lower(), ent.label_) for ent in doc2.ents])

    overlap = ne_types1.intersection(ne_types2)
    union = ne_types1.union(ne_types2)
    return len(overlap) / len(union) if len(union) else 0

def simplified_lesk_similarity(tokens1, tokens2):
    """
    Simplified Lesk algorithm: count overlapping words in definitions (glosses) of the words.
    """
    glosses1 = []
    for token in tokens1:
        synsets = wn.synsets(token)
        for syn in synsets:
            glosses1.extend(syn.definition().split())

    glosses2 = []
    for token in tokens2:
        synsets = wn.synsets(token)
        for syn in synsets:
            glosses2.extend(syn.definition().split())

    overlap = set(glosses1).intersection(set(glosses2))
    union = set(glosses1).union(set(glosses2))
    return len(overlap) / len(union) if len(union) else 0

def hypernym_hyponym_overlap_ratio(tokens1, tokens2):
    """
    Calculate the ratio of words that have hypernyms or hyponyms in common.
    """
    hypernyms1 = set()
    hyponyms1 = set()
    for token in tokens1:
        synsets = wn.synsets(token)
        for syn in synsets:
            hypernyms1.update(set(chain.from_iterable([h.lemma_names() for h in syn.hypernyms()])))
            hyponyms1.update(set(chain.from_iterable([h.lemma_names() for h in syn.hyponyms()])))

    hypernyms2 = set()
    hyponyms2 = set()
    for token in tokens2:
        synsets = wn.synsets(token)
        for syn in synsets:
            hypernyms2.update(set(chain.from_iterable([h.lemma_names() for h in syn.hypernyms()])))
            hyponyms2.update(set(chain.from_iterable([h.lemma_names() for h in syn.hyponyms()])))

    hypernym_overlap = hypernyms1.intersection(hypernyms2)
    hyponym_overlap = hyponyms1.intersection(hyponyms2)
    total_overlap = hypernym_overlap.union(hyponym_overlap)
    total_union = hypernyms1.union(hypernyms2).union(hyponyms1).union(hyponyms2)
    return len(total_overlap) / len(total_union) if len(total_union) else 0

def lsa_similarity(s1, s2):
    """
    Calculate semantic similarity using Latent Semantic Analysis (LSA).
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([s1, s2])
    n_features = tfidf_matrix.shape[1]

    if n_features >= 2:  # Ensure there are enough features for meaningful analysis
        n_components = min(100, n_features - 1)
        try:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            lsa = make_pipeline(svd, Normalizer(copy=False))
            lsa_matrix = lsa.fit_transform(tfidf_matrix)
            sim = cosine_similarity([lsa_matrix[0]], [lsa_matrix[1]])[0][0]
            if np.isnan(sim):  # Handle NaN results
                sim = 0.0
        except ValueError as e:
            # Catch errors like empty matrices or insufficient rank
            sim = 0.0
    else:
        # Fallback to cosine similarity of TF-IDF vectors
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        if np.isnan(sim):  # Handle NaN results
            sim = 0.0
    return sim

def lda_similarity(s1, s2):
    """
    Calculate topic similarity using Latent Dirichlet Allocation (LDA).
    """
    count_vectorizer = CountVectorizer()
    tf_matrix = count_vectorizer.fit_transform([s1, s2])
    n_features = tf_matrix.shape[1]

    if n_features >= 2:
        n_components = min(10, n_features - 1)
        lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
        lda_matrix = lda.fit_transform(tf_matrix)
        sim = cosine_similarity([lda_matrix[0]], [lda_matrix[1]])[0][0]
        if np.isnan(sim):
            sim = 0.0
    else:
        # Not enough features to perform LDA
        sim = 0.0
    return sim

def longest_common_subsequence(X, Y):
    """
    Compute the length of the longest common subsequence between two lists.

    Args:
        X (list): First list.
        Y (list): Second list.

    Returns:
        int: Length of the longest common subsequence.
    """
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                L[i + 1][j + 1] = L[i][j] + 1
            else:
                L[i + 1][j + 1] = max(L[i + 1][j], L[i][j + 1])
    return L[m][n]

def longest_common_substring_length(s1, s2):
    """
    Compute the length of the longest common substring between two strings.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        int: Length of the longest common substring.
    """
    m = len(s1)
    n = len(s2)
    result = 0
    LCSuff = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                LCSuff[i + 1][j + 1] = LCSuff[i][j] + 1
                result = max(result, LCSuff[i + 1][j + 1])
            else:
                LCSuff[i + 1][j + 1] = 0
    return result

def ngram_overlap_ratio(tokens1, tokens2, n=2):
    """
    Calculate n-gram overlap ratio between two token lists.

    Args:
        tokens1 (list): First token list.
        tokens2 (list): Second token list.
        n (int): n-gram size.

    Returns:
        float: n-gram overlap ratio.
    """
    ngrams1 = set(ngrams(tokens1, n))
    ngrams2 = set(ngrams(tokens2, n))
    overlap = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(overlap) / len(union) if len(union) != 0 else 0

def word_order_similarity(tokens1, tokens2):
    """
    Calculate word order similarity between two token lists using Spearman's rank correlation.

    Args:
        tokens1 (list): First token list.
        tokens2 (list): Second token list.

    Returns:
        float: Word order similarity score.
    """
    # Get the union of tokens
    all_tokens = list(set(tokens1 + tokens2))
    # Create index mapping based on the combined tokens
    token_idx = {token: idx for idx, token in enumerate(all_tokens)}

    # Create order vectors for both sentences based on the combined token index
    order_vec1 = [token_idx[token] for token in tokens1 if token in token_idx]
    order_vec2 = [token_idx[token] for token in tokens2 if token in token_idx]

    # Pad the shorter vector to match lengths
    max_len = max(len(order_vec1), len(order_vec2))
    if len(order_vec1) < max_len:
        order_vec1 += [-1] * (max_len - len(order_vec1))
    if len(order_vec2) < max_len:
        order_vec2 += [-1] * (max_len - len(order_vec2))

    # Use Spearman correlation
    if len(order_vec1) >= 2 and len(order_vec2) >= 2:
        correlation, _ = spearmanr(order_vec1, order_vec2)
        return correlation if not np.isnan(correlation) else 0
    else:
        return 0

def tree_edit_distance(s1, s2):
    """
    Compute a simplified tree edit distance between the dependency parse trees of two sentences.

    Args:
        s1 (str): First sentence.
        s2 (str): Second sentence.

    Returns:
        float: Normalized tree edit distance.
    """
    # Reference: Used by [Štajner et al., 2012]
    def get_dependency_tree(sentence):
        doc = nlp(sentence)
        edges = []
        for token in doc:
            for child in token.children:
                edges.append((token.orth_, child.orth_))
        return edges

    tree1 = get_dependency_tree(s1)
    tree2 = get_dependency_tree(s2)

    # Using set difference as a simplification
    diff = len(set(tree1) ^ set(tree2))
    total = len(set(tree1) | set(tree2))
    return diff / total if total != 0 else 0

def antonym_overlap(tokens1, tokens2):
    """
    Compute the ratio of antonym pairs between two lists of tokens.

    Reference: Inspired by [Bär et al., 2012]

    Returns:
        float: Antonym overlap ratio.
    """
    antonym_pairs = 0
    total_pairs = 0
    for word1 in tokens1:
        synsets1 = wn.synsets(word1)
        antonyms1 = set()
        for syn in synsets1:
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    antonyms1.add(ant.name())
        for word2 in tokens2:
            if word2 in antonyms1:
                antonym_pairs += 1
            total_pairs += 1
    return antonym_pairs / total_pairs if total_pairs > 0 else 0

def compute_sentiment_score(tokens):
    """
    Compute average sentiment scores using SentiWordNet.

    Args:
        tokens (list): List of tokens.

    Returns:
        pos_score (float): Positive sentiment score.
        neg_score (float): Negative sentiment score.
        obj_score (float): Objective sentiment score.
    """
    pos_score = 0
    neg_score = 0
    obj_score = 0
    count = 0
    for word in tokens:
        synsets = wn.synsets(word)
        if synsets:
            syn = synsets[0]  # Take the first synset
            swn_synset = swn.senti_synset(syn.name())
            pos_score += swn_synset.pos_score()
            neg_score += swn_synset.neg_score()
            obj_score += swn_synset.obj_score()
            count += 1
    if count > 0:
        pos_score /= count
        neg_score /= count
        obj_score /= count
    return pos_score, neg_score, obj_score

def count_negations(tokens):
    """
    Count the number of negation words in a list of tokens.

    Args:
        tokens (list): List of tokens.

    Returns:
        int: Number of negation words.
    """
    negations = set(["not", "no", "never", "n't", "nothing", "nowhere", "neither", "without", "hardly", "barely", "scarcely"])
    return sum(1 for token in tokens if token.lower() in negations)

def soft_jaccard_similarity(tokens1, tokens2):
    """
    Compute the Soft Jaccard similarity between two sets of tokens.

    Reference: [Jimenez et al., 2012] - UNAL-NLP: Combining Soft Cardinality Features
    """
    intersection = 0
    union = 0
    for token1 in tokens1:
        max_sim = max([word_similarity(token1, token2) for token2 in tokens2])
        intersection += max_sim
    for token in tokens1.union(tokens2):
        union += 1
    return intersection / (union - intersection) if union != intersection else 1

def word_similarity(w1, w2):
    """
    Compute semantic similarity between two words using WordNet.

    Returns:
        float: Maximum Wu-Palmer similarity between synsets of w1 and w2.
    """
    synsets1 = wn.synsets(w1)
    synsets2 = wn.synsets(w2)
    max_sim = 0
    for s1 in synsets1:
        for s2 in synsets2:
            sim = s1.wup_similarity(s2)
            if sim is not None and sim > max_sim:
                max_sim = sim
    return max_sim if max_sim is not None else 0

def semantic_role_overlap(s1, s2):
    """
    Compute the overlap of semantic roles between two sentences.

    Reference: [Heilman and Madnani, 2012] - ETS: Discriminative Edit Models
    """
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    srl1 = extract_semantic_roles(doc1)
    srl2 = extract_semantic_roles(doc2)
    overlap = srl1.intersection(srl2)
    union = srl1.union(srl2)
    return len(overlap) / len(union) if len(union) else 0

def extract_semantic_roles(doc):
    """
    Extract simplified semantic roles from a spaCy Doc.
    """
    roles = set()
    for token in doc:
        if token.dep_ in {'nsubj', 'dobj', 'iobj', 'pobj', 'attr'}:
            roles.add((token.lemma_, token.dep_))
    return roles

def temporal_expression_overlap(s1, s2):
    """
    Compute overlap of temporal expressions between two sentences.

    Reference: [Sultan et al., 2012] - DLS@CU: Sentence Similarity from Word Alignment
    """
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    temp_expressions1 = set([ent.text for ent in doc1.ents if ent.label_ in {'DATE', 'TIME'}])
    temp_expressions2 = set([ent.text for ent in doc2.ents if ent.label_ in {'DATE', 'TIME'}])
    overlap = temp_expressions1.intersection(temp_expressions2)
    union = temp_expressions1.union(temp_expressions2)
    return len(overlap) / len(union) if len(union) else 0